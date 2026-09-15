"""
Symbolic game state, read straight out of NES RAM. Zero API cost.

This is the harness's answer to the failure the baseline documented in
config.py: "a still frame carries no scale for converting pixels to
frames". The model judged a Goomba further away than it was and ran into
it. RAM has the exact number, so this module hands it over.

EVERY ADDRESS BELOW WAS CHECKED AGAINST THE LIVE EMULATOR, not copied
from a memory map and trusted. The checks, on SuperMarioBros-1-1-v0:

  0x006D/0x0086  Mario level x (page, px)   == info["x_pos"] exactly
  0x03AD         Mario x on screen          112 once scrolling starts
  0x0057         horizontal speed, int8     48 while running = 3.0 px/frame
                                            (measured 120 px over 40 frames)
  0x001D         float state                0 grounded, 1 jumping,
                                            2 walked off a ledge, 3 flagpole
  0x03B8         Mario y on screen (top-16) 176 on the 1-1 ground
  0x000E         player state               8 normal, 0x0B dying (enemy)
  0x00B5         vertical screen page       1 normal, >1 fell below screen
  0x000F+i       enemy slot i active        5 slots
  0x0016+i       enemy type                 0x06 Goomba in 1-1
  0x006E+i/0x0087+i  enemy level x          Goomba closes at -0.625 px/frame
  0x00CF+i       enemy y on screen
  0x0500-0x069F  metatile buffer            2 pages x 13 rows x 16 cols;
                                            ? blocks and the first pipe land
                                            on exactly the columns the
                                            screenshot shows them in

Screen geometry: tile row r covers screen y = 32 + 16*r (rows 0..12), so
the 1-1 ground is rows 11-12.

LOOKAHEAD IS REAL BUT BOUNDED -- MEASURED, NOT ASSUMED. The buffer is
written a few columns ahead of the scroll. Predictions made for columns
up to 22 past the screen's left tile column matched what those columns
held once they scrolled on screen in 100% of ~100 checks each; from 23
onward they mismatched ~60% of the time, because the two-page buffer
wraps and still holds the previous page. MAX_LOOKAHEAD_COLS = 20 keeps a
two-column margin. Past it, this module reports nothing rather than
something stale.

LIMITATIONS, stated so nobody rediscovers them the hard way:
  * Pits are "no solid tile in the bottom two rows". True for every pit
    in 1-1; a level with platforms over a void would need more.
  * Enemy type names cover what 1-1 contains plus the common ones.
    Anything else is reported as its hex id, never guessed at.
  * Koopa shell state is not decoded -- a shell is still "Koopa".
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ------------------------------------------------------------ addresses

ADDR_MARIO_PAGE = 0x006D
ADDR_MARIO_X = 0x0086
ADDR_MARIO_SCREEN_X = 0x03AD
ADDR_MARIO_Y = 0x03B8
ADDR_SPEED = 0x0057
ADDR_FLOAT = 0x001D
ADDR_PLAYER_STATE = 0x000E
ADDR_Y_VIEWPORT = 0x00B5
ADDR_POWERUP = 0x0756
ADDR_ENEMY_ACTIVE = 0x000F
ADDR_ENEMY_TYPE = 0x0016
ADDR_ENEMY_PAGE = 0x006E
ADDR_ENEMY_X = 0x0087
ADDR_ENEMY_Y = 0x00CF
ADDR_TILES = 0x0500
N_ENEMY_SLOTS = 5
TILE_ROWS = 13
TILE_COLS_PER_PAGE = 16
TILE_ROW0_Y = 32
MAX_LOOKAHEAD_COLS = 20      # from the screen's left tile column; see above

PLAYER_STATE_DYING = 0x0B

ENEMY_NAMES = {
    0x00: "Koopa",
    0x01: "Red Koopa",
    0x02: "Buzzy Beetle",
    0x03: "Red Koopa",
    0x05: "Hammer Bro",
    0x06: "Goomba",
    0x07: "Blooper",
    0x08: "Bullet Bill",
    0x0D: "Piranha Plant",
    0x0E: "Paratroopa",
    0x11: "Lakitu",
    0x12: "Spiny",
    0x2D: "Bowser",
}
# Objects that live in enemy slots but are not threats.
NON_HAZARD_TYPES = {
    0x30: "flagpole flag",
    0x31: "flagpole",
}

# Metatile ids that do NOT block movement. Everything else non-zero is
# treated as solid, which is the safe direction to be wrong in: a phantom
# wall produces a cautious jump, a phantom gap produces a death.
NON_SOLID_TILES = {0x00, 0xC2, 0xC3, 0x24, 0x25}
PIPE_TILES = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15}
QUESTION_TILES = {0xC0, 0xC1}
COIN_TILES = {0xC2, 0xC3}
FLAGPOLE_TILES = {0x24, 0x25}

# Actions (SIMPLE_MOVEMENT index) that press A.
JUMP_ACTIONS = {2, 4, 5}

MARIO_WIDTH = 16


@dataclass
class Enemy:
    slot: int
    type_id: int
    name: str
    x: int              # level x, left edge
    y: int              # screen y
    row: int
    gap_px: int         # px from Mario's front edge to its near edge (+ ahead)
    vx: Optional[float]  # measured px/frame, None if unmeasurable
    hazard: bool
    frames_to_contact: Optional[float] = None


@dataclass
class TerrainFeature:
    kind: str           # "wall" | "pit"
    col: int            # level tile column where it starts
    gap_px: int         # px from Mario's front edge
    size_tiles: int     # wall height or pit width
    frames_to_contact: Optional[float] = None
    is_pipe: bool = False
    truncated: bool = False     # pit runs past the reliable lookahead


@dataclass
class RamState:
    x: int
    screen_x: int
    edge_x: int
    y: int
    row: int
    speed: float        # px/frame, signed, from 0x57/16
    float_state: int
    grounded: bool
    powerup: int
    player_state: int
    dying: bool
    in_pit: bool
    tiles: np.ndarray   # (13, 21) starting at edge_col; cols 17+ are off screen
    edge_col: int
    enemies: list = field(default_factory=list)
    terrain: list = field(default_factory=list)


# --------------------------------------------------------------- tiles

def tile_at(ram: np.ndarray, level_col: int, row: int) -> int:
    """Metatile at a LEVEL column. The buffer holds two pages and wraps."""
    page = (level_col // TILE_COLS_PER_PAGE) % 2
    col = level_col % TILE_COLS_PER_PAGE
    return int(ram[ADDR_TILES + page * TILE_ROWS * TILE_COLS_PER_PAGE
                   + row * TILE_COLS_PER_PAGE + col])


def is_solid(tile: int) -> bool:
    return tile not in NON_SOLID_TILES


def column_surface(ram: np.ndarray, level_col: int) -> Optional[int]:
    """Top row of the solid stack rising from the bottom row, or None.

    None means the bottom row is open: a pit. Floating blocks (? rows)
    are not part of the stack, so they never register as walls.
    """
    if not is_solid(tile_at(ram, level_col, TILE_ROWS - 1)):
        return None
    top = TILE_ROWS - 1
    while top > 0 and is_solid(tile_at(ram, level_col, top - 1)):
        top -= 1
    return top


# ------------------------------------------------------------ reading

def _speed(ram: np.ndarray) -> float:
    return float(np.int8(ram[ADDR_SPEED])) / 16.0


def _enemy_x(ram: np.ndarray, i: int) -> int:
    return int(ram[ADDR_ENEMY_PAGE + i]) * 256 + int(ram[ADDR_ENEMY_X + i])


def read_state(ram: np.ndarray, prev_ram: Optional[np.ndarray] = None,
               frames_between: int = 1) -> RamState:
    """Decode one frame of RAM.

    prev_ram, if given, is RAM from `frames_between` frames earlier and is
    used only to MEASURE enemy velocity. Measured beats assumed: the
    baseline prompt hard-codes Goombas at 0.6 px/frame, which is right
    for Goombas and wrong for everything else.
    """
    x = int(ram[ADDR_MARIO_PAGE]) * 256 + int(ram[ADDR_MARIO_X])
    screen_x = int(ram[ADDR_MARIO_SCREEN_X])
    edge_x = x - screen_x
    y = int(ram[ADDR_MARIO_Y])
    row = (y + 16 - TILE_ROW0_Y) // 16
    speed = _speed(ram)
    float_state = int(ram[ADDR_FLOAT])
    player_state = int(ram[ADDR_PLAYER_STATE])
    y_viewport = int(ram[ADDR_Y_VIEWPORT])

    edge_col = max(0, edge_x // 16)
    tiles = np.zeros((TILE_ROWS, MAX_LOOKAHEAD_COLS + 1), dtype=np.int32)
    for c in range(MAX_LOOKAHEAD_COLS + 1):
        for r in range(TILE_ROWS):
            tiles[r, c] = tile_at(ram, edge_col + c, r)

    st = RamState(
        x=x, screen_x=screen_x, edge_x=edge_x, y=y, row=row, speed=speed,
        float_state=float_state, grounded=(float_state == 0),
        powerup=int(ram[ADDR_POWERUP]), player_state=player_state,
        dying=(player_state == PLAYER_STATE_DYING or y_viewport > 1),
        in_pit=(y_viewport > 1), tiles=tiles, edge_col=edge_col,
    )

    front = x + MARIO_WIDTH
    closing_speed = max(speed, 0.0)

    # ---- enemies
    for i in range(N_ENEMY_SLOTS):
        if not ram[ADDR_ENEMY_ACTIVE + i]:
            continue
        t = int(ram[ADDR_ENEMY_TYPE + i])
        ex = _enemy_x(ram, i)
        ey = int(ram[ADDR_ENEMY_Y + i])
        vx = None
        if (prev_ram is not None and prev_ram[ADDR_ENEMY_ACTIVE + i]
                and int(prev_ram[ADDR_ENEMY_TYPE + i]) == t and frames_between > 0):
            dx = ex - _enemy_x(prev_ram, i)
            if abs(dx) < 8 * frames_between:     # slot reused -> ignore
                vx = dx / frames_between
        if t in NON_HAZARD_TYPES:
            name, hazard = NON_HAZARD_TYPES[t], False
        else:
            name, hazard = ENEMY_NAMES.get(t, f"enemy(type 0x{t:02X})"), True
        e_row = (ey + 8 - TILE_ROW0_Y) // 16
        if ex >= x:
            gap = ex - front
        else:
            gap = (ex + MARIO_WIDTH) - x    # negative: behind Mario
        e = Enemy(slot=i, type_id=t, name=name, x=ex, y=ey,
                  row=e_row, gap_px=gap, vx=vx,
                  hazard=hazard)
        # Contact only means something for an enemy on Mario's level. A
        # Goomba walking on a brick ledge 4 rows up is 15 px ahead and
        # will never touch him; reporting "contact ~4 frames" for it
        # would trigger exactly the panic jump the numbers exist to stop.
        if hazard and gap >= 0 and abs(e_row - row) <= 1:
            closing = closing_speed - (vx if vx is not None else 0.0)
            if closing > 0.2:
                e.frames_to_contact = max(gap, 0) / closing
        st.enemies.append(e)
    st.enemies.sort(key=lambda e: (e.gap_px < 0, abs(e.gap_px)))

    # ---- terrain ahead
    mario_col = (x + 8) // 16
    support = column_surface(ram, mario_col)
    if support is None:
        support = row + 1
    col = mario_col + 1
    last_col = edge_col + MAX_LOOKAHEAD_COLS
    while col <= last_col:
        surf = column_surface(ram, col)
        gap = max(0, col * 16 - front)
        if surf is None:
            width = 0
            while col + width <= last_col and column_surface(ram, col + width) is None:
                width += 1
            f = TerrainFeature("pit", col, gap, width)
            f.truncated = (col + width > last_col)
            st.terrain.append(f)
            col += width
            continue
        if surf < support:
            is_pipe = tile_at(ram, col, surf) in PIPE_TILES
            st.terrain.append(TerrainFeature("wall", col, gap, support - surf,
                                             is_pipe=is_pipe))
            # skip the rest of this obstacle; a staircase reports only
            # its first step, the model sees the rest in the map
            while col <= last_col and (column_surface(ram, col) or 99) < support:
                col += 1
            continue
        col += 1
    if closing_speed > 0.2:
        for f in st.terrain:
            f.frames_to_contact = f.gap_px / closing_speed
    return st


def death_cause(ram: np.ndarray, info: dict) -> str:
    """Why the episode ended, from RAM. Works for every agent, harness or not."""
    if info.get("flag_get"):
        return "flag"
    if int(ram[ADDR_Y_VIEWPORT]) > 1:
        return "fell into pit"
    if int(ram[ADDR_PLAYER_STATE]) == PLAYER_STATE_DYING:
        st = read_state(ram)
        near = [e for e in st.enemies if e.hazard and abs(e.gap_px) <= 24]
        return f"hit by {near[0].name}" if near else "hit by enemy"
    if int(info.get("time", 1)) <= 0:
        return "timer ran out"
    return "unknown"


# ----------------------------------------------------------- rendering

def ascii_map(st: RamState) -> str:
    """13 rows x 21 cols: the visible screen, a ':' divider, then the
    four reliable off-screen columns ahead.

    Without the off-screen part the map and the hazard list disagree: the
    list reports a pit 110 px ahead that the map has no room to show, and
    a model that cross-checks the two concludes one of them is wrong.
    Legend is printed with the map, so the model never has to remember it.
    """
    visible = TILE_COLS_PER_PAGE + 1
    grid = []
    for r in range(TILE_ROWS):
        line = []
        for c in range(st.tiles.shape[1]):
            t = int(st.tiles[r, c])
            if t == 0:
                ch = "."
            elif t in PIPE_TILES:
                ch = "P"
            elif t in QUESTION_TILES:
                ch = "?"
            elif t in COIN_TILES:
                ch = "o"
            elif t in FLAGPOLE_TILES:
                ch = "|"
            else:
                ch = "#"
            line.append(ch)
        grid.append(line)

    def put(level_x, row, ch):
        c = (level_x + 8) // 16 - st.edge_col
        if 0 <= row < TILE_ROWS and 0 <= c < len(grid[0]):
            grid[row][c] = ch

    for e in st.enemies:
        put(e.x, e.row, "E" if e.hazard else "F")
    put(st.x, st.row, "M")
    return "\n".join("".join(line[:visible]) + ":" + "".join(line[visible:])
                     for line in grid)


def _fmt_frames(f: Optional[float]) -> str:
    return f"~{f:.0f} frames" if f is not None else "not closing"


def describe(st: RamState, include_map: bool = True, max_items: int = 4) -> str:
    """The PERCEPTION block the reasoning model reads."""
    lines = [
        f"mario: level_x={st.x}, screen_x={st.screen_x}, "
        f"speed={st.speed:+.2f} px/frame, "
        f"{'GROUNDED' if st.grounded else 'AIRBORNE'}, "
        f"size={'small' if st.powerup == 0 else 'big'}",
    ]
    hazards = []
    for e in st.enemies:
        if not e.hazard:
            if 0 <= e.gap_px <= 256:
                hazards.append((e.gap_px, f"{e.name} (goal) {e.gap_px} px ahead"))
            continue
        if e.gap_px < -24:
            continue
        motion = (f"moving {'left' if e.vx < 0 else 'right'} {abs(e.vx):.2f} px/frame"
                  if e.vx is not None and abs(e.vx) > 0.05 else
                  ("stationary" if e.vx is not None else "velocity unmeasured"))
        height = "at Mario's level" if abs(e.row - st.row) <= 1 else (
            "ABOVE Mario (platform or falling), not on his path yet"
            if e.row < st.row else "below Mario")
        where = (f"{e.gap_px} px ahead" if e.gap_px >= 0
                 else f"overlapping/behind by {-e.gap_px} px")
        hazards.append((e.gap_px,
                        f"{e.name}: {where}, {height}, {motion}"
                        + (f", contact {_fmt_frames(e.frames_to_contact)}"
                           if abs(e.row - st.row) <= 1 else "")))
    for f in st.terrain:
        if f.kind == "pit":
            wide = f"{'at least ' if f.truncated else ''}{f.size_tiles}"
            desc = (f"PIT {wide} tile(s) ({f.size_tiles * 16}{'+' if f.truncated else ''} px) wide: "
                    f"edge {f.gap_px} px ahead, reach edge {_fmt_frames(f.frames_to_contact)}")
        else:
            what = "PIPE" if f.is_pipe else "WALL/STEP"
            desc = (f"{what} {f.size_tiles} tile(s) ({f.size_tiles * 16} px) tall: "
                    f"{f.gap_px} px ahead, contact {_fmt_frames(f.frames_to_contact)}")
        hazards.append((f.gap_px, desc))
    hazards.sort(key=lambda h: h[0])
    if hazards:
        lines.append("hazards, nearest first (distances from Mario's FRONT edge; "
                     "terrain is exact up to ~100 px past the right screen edge):")
        lines += [f"  - {h[1]}" for h in hazards[:max_items]]
    else:
        lines.append("hazards: none on screen ahead -- flat running ground")
    if include_map:
        lines.append("screen map (M=Mario E=enemy #=solid P=pipe ?=block "
                     "o=coin |=flagpole .=air; 1 char = 16 px; bottom rows are ground, "
                     "a '.' gap in them is a pit; columns right of ':' are just "
                     "off screen):")
        lines.append(ascii_map(st))
    return "\n".join(lines)
