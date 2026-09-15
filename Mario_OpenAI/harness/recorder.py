"""
A gym wrapper that remembers the last few frames of RAM (and optionally
screens), so perception can measure motion without stepping the game.

Why a wrapper and not a change to openai_play.run_segment / land: those
functions stay exactly as the baseline runs them, and anything that
calls env.step() -- including them -- now feeds the history for free.

gym 0.25.2 API, same as everywhere else in this project: reset() returns
the observation only, step() returns a 4-tuple.
"""

from collections import deque
from typing import Optional

import gym
import numpy as np


class StateRecorder(gym.Wrapper):

    def __init__(self, env, depth: int = 16, keep_screens: bool = False):
        super().__init__(env)
        self.depth = depth
        self.keep_screens = keep_screens
        self._ram = deque(maxlen=depth)
        self._screens = deque(maxlen=depth)

    def _record(self):
        u = self.env.unwrapped
        # ram and screen are live views into the emulator: copy or every
        # entry in the deque aliases the current frame.
        self._ram.append(np.array(u.ram, copy=True))
        if self.keep_screens:
            self._screens.append(np.array(u.screen, copy=True))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._ram.clear()
        self._screens.clear()
        self._record()
        return obs

    def step(self, action):
        out = self.env.step(action)
        self._record()
        return out

    def ram_ago(self, n: int) -> tuple:
        """(ram from up to n frames ago, how many frames ago it really is)."""
        if not self._ram:
            return None, 0
        k = min(n, len(self._ram) - 1)
        return self._ram[-1 - k], k

    def screen_ago(self, n: int) -> Optional[np.ndarray]:
        if not self._screens:
            return None
        return self._screens[-1 - min(n, len(self._screens) - 1)]
