import gym
import numpy as np

from gym.wrappers import GrayScaleObservation
from gym.wrappers import ResizeObservation
from gym.wrappers import FrameStack

from .config import IMAGE_SIZE, STACK_SIZE, FRAME_SKIP


class SkipFrame(gym.Wrapper):
    """
    Repeat the same action for multiple frames and
    accumulate the reward.

    NOTE: written for the gym 0.25.x 4-tuple step API
    (obs, reward, done, info). If gym is ever upgraded past 0.26,
    step() returns 5 values (terminated/truncated) and this wrapper
    must be updated.
    """

    def __init__(self, env, skip=FRAME_SKIP):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self._skip):
            frame, reward, done, info = self.env.step(action)

            total_reward += reward

            if done:
                break

        return frame, total_reward, done, info


class SqueezeObservation(gym.ObservationWrapper):
    """
    Safety net: if a wrapper upstream produces stacked grayscale
    observations with a trailing channel dimension

        (4, 84, 84, 1)

    squeeze it into

        (4, 84, 84)

    which is the CNN input format. With
    GrayScaleObservation(keep_dim=False), observations are already
    (4, 84, 84) and this wrapper is a pass-through.

    VECTOR-ENV NOTE: this wrapper also materializes LazyFrames into a
    real uint8 array whose dtype/shape MATCH the declared
    observation_space -- the invariant a shared-memory AsyncVectorEnv
    depends on (a mismatch silently zeroes every pixel; see
    vector_env.assert_obs_matches_space).
    """

    def __init__(self, env):
        super().__init__(env)

        old_space = self.env.observation_space
        old_shape = old_space.shape

        if len(old_shape) == 4 and old_shape[-1] == 1:
            new_shape = old_shape[:-1]
        else:
            new_shape = old_shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        obs = np.array(obs, dtype=np.uint8)

        if obs.ndim == 4 and obs.shape[-1] == 1:
            obs = np.squeeze(obs, axis=-1)

        return obs


class NoopResetEnv(gym.Wrapper):
    """Start each episode after a random number of no-op actions.

    WHY: World 1-1 is fully deterministic, so without this the agent
    can reach the flag by memorizing ONE fixed action sequence rather
    than learning a policy. Colab runs showed exactly that signature:
    at eps=0.05 training cleared the flag in 18.6% of episodes, but
    eval at eps=0.02 cleared only ~8%, and eight separate evals died
    at the same x_pos ~310 with all episodes identical. LESS
    randomness giving WORSE results means the greedy trajectory had
    deterministic traps that random actions were breaking by accident.
    Adding this wrapper took the eval flag rate from ~8% to 33% and
    the stall clusters vanished.

    Randomizing the start phase forces the policy to handle states it
    cannot have memorized. EXPECT A TEMPORARY DIP on a policy that had
    memorized one path: it has to generalize before it improves again.

    Placed OUTSIDE FrameStack, so each no-op is one agent step (= 4
    NES frames via SkipFrame) and the stack is filled with post-no-op
    frames. noop_max=0 disables.
    """

    def __init__(self, env, noop_max=30, noop_action=0):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = noop_action

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.noop_max <= 0:
            return obs
        for _ in range(np.random.randint(1, self.noop_max + 1)):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:                      # can't happen at 1-1's start,
                obs = self.env.reset()    # but never return a dead state
                break
        return obs


def build_env(env, noop_max=0):
    """
    Apply the complete Mario preprocessing pipeline.

    Raw RGB frame (uint8):
        (240, 256, 3)

    After grayscale (keep_dim=False):
        (240, 256)

    After resize:
        (84, 84)

    After frame stack:
        (4, 84, 84)

    Frames stay uint8 [0, 255] end-to-end. Normalization to
    float32 [0, 1] is deferred to the agent (recall()/act()) so the
    replay buffer stores compact uint8 frames -- a 4x RAM saving over
    storing float32. Staying uint8 also halves the bytes crossing the
    process boundary per step under AsyncVectorEnv, and keeps the
    observation DATA dtype equal to the DECLARED space dtype (the
    invariant shared-memory vector envs silently corrupt frames
    without).

    noop_max > 0 appends NoopResetEnv (random no-op starts) -- used by
    training and evaluation. play.py records with noop_max=0 so clips
    stay deterministic showcases of the greedy policy.
    """

    env = SkipFrame(env, skip=FRAME_SKIP)

    env = GrayScaleObservation(
        env,
        keep_dim=False
    )

    env = ResizeObservation(
        env,
        shape=IMAGE_SIZE
    )

    env = FrameStack(
        env,
        num_stack=STACK_SIZE
    )

    env = SqueezeObservation(env)

    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)

    return env


def observation_to_numpy(obs):
    """
    Convert LazyFrames into a NumPy array.

    Output:
        (4, 84, 84) uint8
    """

    return np.array(obs, dtype=np.uint8)


def observation_to_tensor(obs, device=None):
    """
    Convert observation into normalized PyTorch format.

    Input:
        (4, 84, 84) uint8 [0, 255]

    Output:
        (1, 4, 84, 84) float32 [0, 1]
    """

    import torch

    tensor = torch.tensor(
        np.array(obs, dtype=np.uint8),
        dtype=torch.float32
    ).unsqueeze(0) / 255.0

    if device is not None:
        tensor = tensor.to(device)

    return tensor
