"""
mario_agent/config.py -- single source of truth for preprocessing shape.

Both the network (dqn_model.py validates its 84x84 input) and the env
pipeline (data_pipeline.py builds skip -> gray -> resize -> stack) read
these constants, and the top-level config.py re-exports them, so the
training scripts and the package can never disagree about frame shapes.

CAUTION: IMAGE_SIZE is baked into the checkpoint architecture -- the
first Linear layer (3136 -> 512) is sized for the 7x7x64 conv output of
an 84x84 input. Changing it invalidates every existing checkpoint.
"""

# Side length of the square grayscale frame fed to the CNN.
IMAGE_SIZE = 84

# Consecutive frames stacked into one observation (= CNN input channels).
STACK_SIZE = 4

# Emulated NES frames repeated per agent action by SkipFrame.
FRAME_SKIP = 4
