"""
mario_agent/vector_env.py -- everything gym.vector needs to work with
this exact stack (gym 0.25.2 x numpy 1.26.4 x nes-py), ported from the
Colab parallel notebook where each fix was reproduced empirically
before being written.

Three bugs live here so train.py doesn't have to know about them:

1. gym 0.25.2 x numpy >= 1.25 crash. Every gym.vector constructor
   deepcopies a space's RNG, and numpy >= 1.25's Generator.__reduce__
   passes an argument gym's _generator_ctor does not accept --
   AsyncVectorEnv cannot even be constructed. numpy==1.26.4 is a hard
   pin (nes-py), so gym is what gets patched. The replacement MUST be
   a module-level function: spaces are pickled when sent to worker
   processes, and a Generator's __reduce__ names this very function.
   A closure would fail worker startup with "Can't pickle local
   object".

2. Observation dtype vs declared space. A vector env with
   shared_memory=True copies observations into a buffer typed from
   observation_space.dtype. If a wrapper changes the DATA dtype
   without updating the DECLARED space (the Colab chain's x/255
   transform did exactly this), np.copyto silently truncates every
   pixel to 0 and training runs forever on black frames. This repo's
   pipeline stays uint8 end-to-end and SqueezeObservation declares
   uint8, so data and declaration agree -- but assert_obs_matches_space
   makes the invariant checkable instead of assumed.

3. Corrupted episode boundaries. Vector envs auto-reset: on a done
   step the returned observation is already the NEXT episode's first
   frame. Caching it as next_state teaches the agent that dying
   teleports it to the level start. The true terminal frame arrives in
   the info dict; parse_vector_infos() digs it out of either info
   layout gym might use.
"""

import inspect

import numpy as np


# ---------------------------------------------------------------------
# 1. gym 0.25.2 x numpy>=1.25 incompatibility (breaks ALL gym.vector use)
# ---------------------------------------------------------------------
_ORIG_GENERATOR_CTOR = None


def _gym_generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=None):
    """Picklable, module-level replacement that drops the extra argument.

    Must return gym's RandomNumberGenerator subclass, because
    Space.__init__ isinstance-checks the seed it is handed.
    """
    return _ORIG_GENERATOR_CTOR(bit_generator_name)


def patch_gym_rng():
    """gym 0.25.2's RandomNumberGenerator._generator_ctor() accepts ONE
    argument; numpy >= 1.25's Generator.__reduce__ passes TWO. Every
    gym.vector constructor deepcopies a space's RNG, so AsyncVectorEnv
    dies at construction with a baffling TypeError about
    _generator_ctor. Returns False (no-op) if a future gym already
    fixed the signature. Idempotent -- safe to call more than once.
    """
    global _ORIG_GENERATOR_CTOR
    from gym.utils.seeding import RandomNumberGenerator as _RNG

    if _ORIG_GENERATOR_CTOR is not None:
        return True  # already patched
    if len(inspect.signature(_RNG._generator_ctor).parameters) >= 2:
        return False
    _ORIG_GENERATOR_CTOR = _RNG._generator_ctor
    _RNG._generator_ctor = staticmethod(_gym_generator_ctor)
    return True


# ---------------------------------------------------------------------
# 2. Declared-space sanity check (the silent-black-frames bug, class of)
# ---------------------------------------------------------------------
def assert_obs_matches_space(env):
    """Raise if a reset observation's dtype/shape disagree with the
    declared observation_space. shared_memory=True vector envs copy
    into a buffer typed from the DECLARED space; a mismatch either
    raises (shared_memory=False) or silently zeroes every pixel
    (shared_memory=True) -- both were reproduced on Colab. Cheap: one
    reset on one env, called once at startup.
    """
    obs = np.asarray(env.reset())
    space = env.observation_space
    if obs.dtype != space.dtype or tuple(obs.shape) != tuple(space.shape):
        raise RuntimeError(
            f"Observation ({obs.dtype}, {obs.shape}) does not match the "
            f"declared observation_space ({space.dtype}, {space.shape}). "
            f"A shared-memory vector env would silently corrupt every "
            f"frame. Fix the wrapper that changed the data without "
            f"updating the declared space."
        )


# ---------------------------------------------------------------------
# 3. Info parsing + terminal-frame recovery
# ---------------------------------------------------------------------
def parse_vector_infos(infos, num_envs):
    """gym 0.25.2 vector envs return infos as a DICT OF ARRAYS with a
    boolean `_key` mask per key -- not the list of dicts a single env
    gives -- and the terminal frame arrives under `final_observation`.

    Returns (per_env_info_dicts, final_obs), where final_obs[i] is the
    TRUE last frame of env i's episode or None if it did not finish
    this step. Also handles a list-of-dicts layout, in case of a
    version change.
    """
    per_env = [{} for _ in range(num_envs)]
    final_obs = [None] * num_envs
    if isinstance(infos, dict):
        for key, values in infos.items():
            if key.startswith("_"):
                continue
            mask = infos.get("_" + key)
            for i in range(num_envs):
                if mask is None or bool(mask[i]):
                    if key == "final_observation":
                        final_obs[i] = values[i]
                    else:
                        per_env[i][key] = values[i]
    else:
        for i, info in enumerate(list(infos)[:num_envs]):
            info = dict(info or {})
            fo = info.pop("final_observation", None)
            if fo is None:
                fo = info.pop("terminal_observation", None)
            final_obs[i] = fo
            per_env[i] = info
    return per_env, final_obs


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def make_vec_env(env_fn, n_envs, shared_memory=True, check_spaces=True):
    """Patch gym's RNG, optionally verify the declared observation
    space against real data, then launch n_envs worker processes.

    ORDERING NOTE FOR CALLERS: call this BEFORE allocating the replay
    buffer (and before any CUDA work). fork() gives children a
    copy-on-write view of the parent; the parent rewrites the multi-GB
    buffer as the ring cycles, so forking after allocation eventually
    DOUBLES its physical footprint (measured: 1409 MB vs 662 MB on a
    700 MB stand-in). Forking first also creates the children before
    the CUDA context exists, the officially safe ordering.
    """
    patch_gym_rng()
    from gym.vector import AsyncVectorEnv

    if check_spaces:
        probe = env_fn()
        try:
            assert_obs_matches_space(probe)
        finally:
            probe.close()

    return AsyncVectorEnv([env_fn for _ in range(n_envs)],
                          shared_memory=shared_memory)
