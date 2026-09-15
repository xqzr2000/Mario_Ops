"""
Image scaffolding: annotations drawn onto the screenshot before it is
sent. GamingAgent calls this "scaffolding" and its Mario config uses a
5x5 labelled grid; that is `grid` here. `ruler` is new.

Drawn AFTER the nearest-neighbour upscale. On the native 256x240 frame
a label is 5 px tall and unreadable; at 3x it is legible and the sprite
pixels underneath are untouched.

With scaffold "none" the image comes from OpenAIMarioAgent.encode_screen
itself, not a reimplementation, so the baseline preset sends exactly the
bytes the baseline sends.
"""

import base64

import cv2
import numpy as np

import config
from openai_agent import OpenAIMarioAgent


def _upscale(screen_rgb: np.ndarray) -> np.ndarray:
    if config.SCREEN_UPSCALE > 1:
        return cv2.resize(screen_rgb, None, fx=config.SCREEN_UPSCALE,
                          fy=config.SCREEN_UPSCALE, interpolation=cv2.INTER_NEAREST)
    return screen_rgb.copy()


def _png_b64(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _label(img, text, org, scale, color):
    """Text with a dark outline so it reads against sky, brick and pipe."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                1, cv2.LINE_AA)


def draw_grid(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """GamingAgent's draw_grid_on_image: cell lines plus (col,row) labels."""
    out = img.copy()
    h, w = out.shape[:2]
    ch, cw = h // rows, w // cols
    for r in range(rows + 1):
        cv2.line(out, (0, min(r * ch, h - 1)), (w, min(r * ch, h - 1)), (0, 255, 0), 2)
    for c in range(cols + 1):
        cv2.line(out, (min(c * cw, w - 1), 0), (min(c * cw, w - 1), h), (0, 255, 0), 2)
    scale = 0.35 * config.SCREEN_UPSCALE / 1.5
    for r in range(rows):
        for c in range(cols):
            _label(out, f"({c},{r})", (c * cw + 6, r * ch + int(18 * scale / 0.7)),
                   scale, (255, 255, 255))
    return out


def draw_ruler(img: np.ndarray, mario_screen_x: int, mario_y: int,
               speed: float, step_px: int) -> np.ndarray:
    """Distance ticks ahead of Mario, labelled in px and frames.

    The baseline's diagnosed failure is a unit conversion: the model sees a
    Goomba "near" and cannot turn near into frames. The ruler puts the
    conversion on the image, next to the thing being judged, at the
    current measured speed.
    """
    k = config.SCREEN_UPSCALE
    out = img.copy()
    h, w = out.shape[:2]
    front = mario_screen_x + 16
    top = max(0, (mario_y + 16 - 40) * k)       # a little above Mario's head
    ground_top = 208 * k                         # tile row 11: the ground band
    scale = 0.17 * k

    overlay = out.copy()
    cv2.line(overlay, (front * k, top), (front * k, ground_top), (0, 255, 255), 2)
    d = step_px
    while front + d < 256:
        x = (front + d) * k
        cv2.line(overlay, (x, top), (x, ground_top), (255, 230, 0), 2)
        d += step_px
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)

    # Labels go IN the ground band. Nothing that matters moves there --
    # enemies, pipes and Mario are all above it -- so the text can be large
    # without covering the thing the ruler is supposed to measure.
    d = step_px
    while front + d < 256:
        x = (front + d) * k
        _label(out, f"{d}px", (x - 14 * k // 3, ground_top + 12 * k), scale, (255, 230, 0))
        if speed > 0.5:
            _label(out, f"{d / speed:.0f}f", (x - 10 * k // 3, ground_top + 24 * k),
                   scale, (255, 255, 255))
        d += step_px
    return out


def encode_for_reasoning(screen_rgb: np.ndarray, scaffold: str, grid: str = "5x5",
                         ram_state=None, step_px: int = 32) -> str:
    if scaffold == "none":
        return OpenAIMarioAgent.encode_screen(screen_rgb)
    img = _upscale(screen_rgb)
    if scaffold == "grid":
        rows, cols = (int(v) for v in grid.lower().split("x"))
        img = draw_grid(img, rows, cols)
    elif scaffold == "ruler":
        if ram_state is None:
            raise ValueError("ruler scaffold needs a RAM state for Mario's screen position")
        img = draw_ruler(img, ram_state.screen_x, ram_state.y,
                         max(ram_state.speed, 0.0), step_px)
    else:
        raise ValueError(f"unknown scaffold {scaffold!r}")
    return _png_b64(img)


def encode_plain(screen_rgb: np.ndarray) -> str:
    return OpenAIMarioAgent.encode_screen(screen_rgb)
