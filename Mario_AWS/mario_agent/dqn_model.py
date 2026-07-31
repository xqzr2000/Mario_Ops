import copy
import torch
import torch.nn as nn

from .config import IMAGE_SIZE


class MarioNet(nn.Module):
    """
    Deep Q-Network used for Double DQN.

    Architecture:
        Input (4, 84, 84)
            ↓
        Conv2D(32, 8x8, stride=4)
            ↓
        ReLU
            ↓
        Conv2D(64, 4x4, stride=2)
            ↓
        ReLU
            ↓
        Conv2D(64, 3x3, stride=1)
            ↓
        ReLU
            ↓
        Flatten
            ↓
        Linear(3136 → 512)
            ↓
        ReLU
            ↓
        Linear(512 → num_actions)

    Two identical networks are maintained:

        self.online
            - Learns from gradient descent.

        self.target
            - Used to compute stable TD targets.
            - Periodically synchronized from online.
            - Frozen (no gradients).
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        channels, height, width = input_dim

        if height != IMAGE_SIZE:
            raise ValueError(
                f"Expected height={IMAGE_SIZE}, got {height}"
            )

        if width != IMAGE_SIZE:
            raise ValueError(
                f"Expected width={IMAGE_SIZE}, got {width}"
            )

        self.online = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(3136, 512),
            nn.ReLU(),

            nn.Linear(512, output_dim)
        )

        # Create target network
        self.target = copy.deepcopy(self.online)

        # Freeze target network parameters
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x, model="online"):
        """
        Forward pass through either network.

        Args:
            x:
                Tensor of shape
                (batch_size, 4, 84, 84)

            model:
                "online" or "target"

        Returns:
            Q-values of shape
            (batch_size, num_actions)
        """

        if model == "online":
            return self.online(x)

        if model == "target":
            return self.target(x)

        raise ValueError(
            f"model must be 'online' or 'target', got '{model}'"
        )

    def sync_target(self):
        """
        Copy online network weights to target network.

        Typically called every N training steps.
        """

        self.target.load_state_dict(
            self.online.state_dict()
        )

    @property
    def online_parameters(self):
        """
        Convenience accessor for optimizer creation.
        """

        return self.online.parameters()
