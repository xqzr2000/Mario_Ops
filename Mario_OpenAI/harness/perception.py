"""
Perception module: screen + RAM in, a Percept out.

GamingAgent's PerceptionModule has two tracks -- edit the image
(scaffolding), then optionally ask a VLM to describe it. Both are here,
plus a third track GamingAgent uses for its non-retro games but not for
Mario: a symbolic state read from the game itself (harness/ram_state.py).

RAM IS ALWAYS READ, EVEN WHEN IT IS NOT SHOWN. With perception "off" or
"vision" the model never sees it, but the loop still uses it for the
death cause in summary.json and for the guard. Harness-off runs get
better post-mortems for free, and the comparison stays fair because
nothing the model receives changes.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from harness import prompts
from harness import settings as S
from harness.ram_state import RamState, describe, read_state
from harness.scaffold import encode_for_reasoning, encode_plain


@dataclass
class Percept:
    ram: RamState
    images: list                      # base64 PNGs for the reasoning call
    ram_text: Optional[str] = None    # PERCEPTION block, if shown
    vision_text: Optional[str] = None  # VISUAL ANALYSIS block, if shown
    extras: dict = field(default_factory=dict)


class PerceptionModule:

    def __init__(self, llm, recorder):
        self.llm = llm
        self.rec = recorder

    def perceive(self) -> Percept:
        ram_now, _ = self.rec.ram_ago(0)
        ram_then, gap = self.rec.ram_ago(S.VELOCITY_WINDOW)
        st = read_state(ram_now, ram_then if gap > 0 else None, max(gap, 1))
        screen = np.array(self.rec.unwrapped.screen, copy=True)

        images = [encode_for_reasoning(screen, S.SCAFFOLD, S.GRID, st, S.RULER_STEP_PX)]
        if S.MOTION_FRAMES > 0:
            earlier = self.rec.screen_ago(S.MOTION_FRAMES)
            if earlier is not None:
                images.append(encode_plain(earlier))

        p = Percept(ram=st, images=images)
        if S.PERCEPTION in ("ram", "both"):
            p.ram_text = describe(st, include_map=S.ASCII_MAP)
        if S.PERCEPTION in ("vision", "both"):
            p.vision_text = self._vision(screen, st)
        return p

    def _vision(self, screen: np.ndarray, st: RamState) -> str:
        """GamingAgent's perception call: describe the GRIDDED image as JSON.

        Always gridded, regardless of HARNESS_SCAFFOLD, because the JSON
        schema is in grid cells -- that is the GamingAgent design being
        reproduced, and a cell reference without the grid is meaningless.
        """
        img = encode_for_reasoning(screen, "grid", S.GRID)
        text = self.llm.complete(
            "perception",
            prompts.VISION_PERCEPTION_SYSTEM,
            prompts.vision_perception_prompt(S.GRID),
            images=[img],
            json_mode=True,
        )
        return text.strip() or "(perception model returned nothing)"
