# mario_agent.py

import random
from pathlib import Path
import numpy as np
import torch

from .dqn_model import MarioNet


def first_if_tuple(x):
    """
    Some Gym versions return (obs, info) tuples from reset()/step().
    This unwraps the observation if a tuple slipped through, otherwise
    returns x unchanged. Guards act() and cache() against version drift.
    """
    return x[0] if isinstance(x, tuple) else x


def resolve_device(device=None):
    """
    Pick the compute device.

        None or "auto"  -> "cuda" if a GPU is visible, else "cpu"
        "cpu" / "cuda"  -> forced

    Local CPU test:     MarioAgent(..., device="cpu")
    AWS GPU instance:   MarioAgent(..., device="auto")  # finds the GPU
    """
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def checkpoint_action_dim(state_dict):
    """
    Read the action count baked into a checkpoint's final layer.

    The output layer's shape IS the action-set lineage: a 7-action
    (SIMPLE_MOVEMENT) checkpoint has a (7, 512) final Linear weight, the
    old 2-action family has (2, 512). Mixing them produces a cryptic
    torch size-mismatch mid-load; checking the shape FIRST turns that
    into a clear one-line error. Returns None if no final layer is
    found (unknown schema -- let load_state_dict report it).
    """
    best_idx, best_key = -1, None
    for key in state_dict:
        if key.startswith("online.") and key.endswith(".weight"):
            try:
                idx = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            if idx > best_idx:
                best_idx, best_key = idx, key
    if best_key is None:
        return None
    weight = state_dict[best_key]
    return int(weight.shape[0]) if hasattr(weight, "shape") else None


class RingReplay:
    """
    Replay memory as PREALLOCATED numpy ring buffers.

    WHY NOT A DEQUE? A deque of per-transition arrays allocates a small
    LONG-LIVED block on every environment step, interleaved on the heap
    with the multi-MB SHORT-LIVED batch temporaries that learning creates
    every few steps. glibc cannot reuse or release memory around those
    interleaved live blocks, so process RSS grows without bound even
    though nothing is leaked in the Python sense. Observed in practice:
    ~7 MB of host RAM lost per learn call, climbing to an 11 GB plateau
    while the buffer itself held under 1 GB.

    Preallocating fixed arrays once means the steady-state training loop
    performs ZERO heap allocation for replay storage: add() is an
    in-place slot write, sample_into_staging() gathers into reusable
    batch arrays. The footprint is paid up front, printed at startup,
    and never changes for the life of the run.
    """

    def __init__(self, capacity, state_shape=(4, 84, 84), batch_size=32):
        self.maxlen = capacity
        self.states = np.zeros((capacity,) + state_shape, dtype=np.uint8)
        self.next_states = np.zeros((capacity,) + state_shape, dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        # Pre-touch every page now so the FULL footprint is resident from
        # startup: the RAM cost is visible immediately (fail fast if the
        # host can't fit it) and RSS stays flat for the rest of the run.
        for arr in (self.states, self.next_states):
            arr.fill(0)

        self.idx = 0
        self.size = 0

        # Reusable staging arrays for sampled batches (allocated once).
        self._bs = np.empty((batch_size,) + state_shape, dtype=np.uint8)
        self._bns = np.empty((batch_size,) + state_shape, dtype=np.uint8)
        # Reusable scratch for float->uint8 frame conversion.
        self._scratch = np.empty(state_shape, dtype=np.float64)
        # Batched-write staging (allocated lazily on first add_batch).
        self._ab_cap = 0

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return (
            self.states[i],
            self.next_states[i],
            self.actions[i],
            self.rewards[i],
            self.dones[i],
        )

    @property
    def nbytes(self):
        return (
            self.states.nbytes
            + self.next_states.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.dones.nbytes
        )

    def _write_frame(self, dest, obs):
        """
        Write one observation into a uint8 slot, in place.

        Handles both frame conventions found across our env builds:
          * uint8 [0, 255]  (raw frames)            -> stored directly
          * float [0, 1]    (x/255. TransformObservation) -> rescaled
        Dtype-based dispatch is safe here because the only source of
        float frames in this pipeline is the /255 normalization wrapper.
        (Casting float [0,1] straight to uint8 would silently truncate
        every pixel to 0 -- the buffer would train on black screens.)
        """
        arr = np.asarray(obs)
        if np.issubdtype(arr.dtype, np.floating):
            np.multiply(arr, 255.0, out=self._scratch)
            dest[self.idx] = self._scratch  # float64 -> uint8 cast on assign
        else:
            dest[self.idx] = arr

    def add(self, state, next_state, action, reward, done):
        """In-place, allocation-free write into the current ring slot."""
        self._write_frame(self.states, state)
        self._write_frame(self.next_states, next_state)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

    def _write_frames_batch(self, dest, src, idxs, n):
        """Batched slot write with the same dtype dispatch as
        _write_frame(). uint8 frames (this repo's pipeline) are written
        directly; float [0,1] frames (Colab-style x/255 chains) are
        rescaled once for the whole block through preallocated staging,
        so the steady-state loop still performs zero heap allocation --
        the entire reason RingReplay exists."""
        src = np.asarray(src)
        if np.issubdtype(src.dtype, np.floating):
            if self._ab_cap < n:
                shape = self.states.shape[1:]
                self._ab_f32 = np.empty((n,) + shape, dtype=np.float32)
                self._ab_u8 = np.empty((n,) + shape, dtype=np.uint8)
                self._ab_cap = n
            f32, u8 = self._ab_f32[:n], self._ab_u8[:n]
            np.multiply(src.astype(np.float32, copy=False), 255.0, out=f32)
            np.copyto(u8, f32, casting="unsafe")
            dest[idxs] = u8
        else:
            dest[idxs] = src

    def add_batch(self, states, next_states, actions, rewards, dones):
        """Write N transitions in one call. Identical slot semantics to
        N sequential add() calls (verified element-for-element on
        Colab), but the frame conversion happens once per block."""
        n = len(actions)
        idxs = (self.idx + np.arange(n)) % self.maxlen
        self._write_frames_batch(self.states, states, idxs, n)
        self._write_frames_batch(self.next_states, next_states, idxs, n)
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones
        self.idx = int((self.idx + n) % self.maxlen)
        self.size = int(min(self.size + n, self.maxlen))

    def sample_into_staging(self, batch_size):
        """Gather a random batch into the preallocated staging arrays."""
        ind = np.random.randint(0, self.size, size=batch_size)
        np.take(self.states, ind, axis=0, out=self._bs)
        np.take(self.next_states, ind, axis=0, out=self._bns)
        return self._bs, self._bns, self.actions[ind], self.rewards[ind], self.dones[ind]

    def clear(self):
        self.idx = 0
        self.size = 0


class MarioFrame:
    def __init__(self, state_dim, action_dim, save_dir, device=None,
                 replay_capacity=100000):
        """
        Initialize the Mario agent base configuration.

        DEVICE NOTE: the network and all learning math live on
        `self.device` (CPU locally, CUDA on a GPU instance). The replay
        buffer deliberately stays on the CPU as uint8 -- storing it on
        the GPU or as float32 would waste VRAM/RAM for no benefit. Only
        the sampled mini-batches are moved to the device, in recall().
        """
        # Environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Compute device (CPU for local tests, CUDA on AWS GPU instances)
        self.device = resolve_device(device)

        # Directory used for saving model checkpoints (ensured as Path object)
        self.save_dir = Path(save_dir) if save_dir else None

        # Deep Q-Network (DQN) used to estimate Q-values
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(self.device)

        # Epsilon-greedy exploration parameters.
        # DECAY NOTE: the tutorial's 0.99999975 keeps the agent >90%
        # random for ~9.2M steps (entire multi-hour sessions with no
        # learning, observed in real training logs). 0.9999975 reaches
        # the floor in ~0.9M steps and is what the level-clearing agent
        # was trained with. FLOOR 0.02: ~11 random actions per episode
        # (at the old 0.05) was free insurance at a 1% flag rate and
        # actively destroys good runs at ~37%. configure_agent() can
        # still override both.
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.02

        # Total number of environment steps taken
        self.curr_step = 0

        # Save the model every 500,000 steps
        self.save_every = int(5e5)

        # Number of experiences sampled from the replay buffer during updates
        self.batch_size = 64

        # Replay buffer for storing past experiences.
        #
        # MEMORY NOTE: frames are stored as uint8 (raw [0, 255] pixels)
        # in PREALLOCATED CPU ring buffers, and only converted to
        # normalized float32 on the compute device when sampled.
        #   * uint8 vs float32:  ~56 KB vs ~226 KB per transition
        #     (at 100k transitions: ~5.6 GB vs ~22 GB)
        #   * preallocated ring vs deque: fixes unbounded RSS growth from
        #     heap fragmentation -- see the RingReplay docstring.
        #
        # FORK-ORDERING NOTE: when training with AsyncVectorEnv, launch
        # the worker processes BEFORE constructing the agent. The
        # parent rewrites this entire buffer as the ring cycles, and
        # copy-on-write duplicates every shared page it touches --
        # forking after allocation roughly doubles the buffer's
        # physical footprint (measured 1409 MB vs 662 MB on a 700 MB
        # stand-in). train.py enforces this ordering.
        self.memory = RingReplay(
            capacity=replay_capacity,
            state_shape=tuple(state_dim),
            batch_size=self.batch_size,
        )

        # Reusable PINNED staging tensors for fast, allocation-free
        # host-to-device copies in recall(). Only these two small
        # (~0.9 MB) tensors are pinned -- never the whole buffer.
        pin = self.device.type == "cuda"
        self._t_s = torch.empty(
            (self.batch_size,) + tuple(state_dim), dtype=torch.uint8, pin_memory=pin
        )
        self._t_ns = torch.empty(
            (self.batch_size,) + tuple(state_dim), dtype=torch.uint8, pin_memory=pin
        )

    @staticmethod
    def to_float_state(state_uint8):
        """
        Convert a uint8 state tensor [0, 255] into a normalized
        float32 tensor [0, 1]. Used at sample/inference time so the
        replay buffer can stay uint8.
        """
        return state_uint8.float() / 255.0

    def _states_to_device(self, states):
        """
        Batch of observations -> normalized float32 tensor on the
        compute device. Dtype dispatch matches the rest of the agent:
        uint8 [0,255] frames (this repo's pipeline) are normalized on
        the device; float [0,1] frames (Colab-style x/255 chains) pass
        straight through as float32.
        """
        arr = np.asarray(states)
        if np.issubdtype(arr.dtype, np.floating):
            return torch.as_tensor(
                np.asarray(arr, dtype=np.float32), device=self.device
            )
        t = torch.as_tensor(arr, device=self.device)
        return t.float().div_(255.0)

    def act(self, state):
        pass

    def cache(self, state, next_state, action, reward, done):
        pass

    def recall(self):
        pass

    def learn(self):
        pass


class MarioAct(MarioFrame):
    def __init__(self, state_dim, action_dim, save_dir, device=None,
                 replay_capacity=100000):
        super().__init__(state_dim, action_dim, save_dir, device, replay_capacity)

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update step value.
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = np.asarray(first_if_tuple(state))
            state = self._states_to_device(state).unsqueeze(0)

            # Inference only -- no need to build a computation graph.
            with torch.no_grad():
                action_values = self.net(state, model="online")

            action_idx = torch.argmax(action_values, dim=1).item()

        # Decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # Increment step counter
        self.curr_step += 1

        return action_idx

    def act_batch(self, states):
        """Epsilon-greedy for N states in ONE forward pass (parallel loop).

        This is a win even at N=2: a batch-1 conv forward is
        launch-latency bound, not compute bound, so a batch-N forward
        costs almost exactly the same -- N-1 of every N forwards become
        free.
        """
        states = np.asarray(states)
        n = len(states)
        actions = np.random.randint(self.action_dim, size=n)
        explore = np.random.rand(n) < self.exploration_rate
        exploit = ~explore
        if exploit.any():
            with torch.no_grad():
                s = self._states_to_device(states[exploit])
                actions[exploit] = (
                    self.net(s, model="online").argmax(dim=1).cpu().numpy()
                )

        # Decay once PER TRANSITION COLLECTED, exactly as the serial
        # act() does: decay**n is the same schedule as n sequential
        # multiplies. The schedule is defined over experience, so N
        # actors reach a given epsilon after the same number of
        # transitions -- just in 1/N the wall-clock time. Nothing to
        # re-tune.
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * (self.exploration_rate_decay ** n),
        )
        self.curr_step += n
        return actions


class MarioCache(MarioAct):
    def __init__(self, state_dim, action_dim, save_dir, device=None,
                 replay_capacity=100000):
        super().__init__(state_dim, action_dim, save_dir, device, replay_capacity)

    def cache(self, state, next_state, action, reward, done):
        """
        Store experience elements into replay memory.

        Frames are written IN PLACE into the preallocated uint8 ring
        buffer on the CPU -- no per-step allocations, no fragmentation.
        Normalization to float32 and the move to the compute device
        happen in recall().
        """
        state = first_if_tuple(state)
        next_state = first_if_tuple(next_state)
        self.memory.add(state, next_state, int(action), float(reward), bool(done))

    def cache_batch(self, states, next_states, actions, rewards, dones):
        """Store N transitions from the vectorized loop in one call."""
        self.memory.add_batch(
            states,
            next_states,
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
        )


class MarioRecall(MarioCache):
    def __init__(self, state_dim, action_dim, save_dir, device=None,
                 replay_capacity=100000):
        super().__init__(state_dim, action_dim, save_dir, device, replay_capacity)

    def recall(self):
        """
        Randomly sample a batch of experiences from replay memory.

        The copy path is allocation-free on the host: ring buffer ->
        preallocated numpy staging -> reusable pinned tensors -> device.
        States come out normalized as float32 [0, 1] on the compute
        device, ready for the CNN; device-side batch tensors are
        recycled by torch's caching allocator.
        """
        bs, bns, action, reward, done = self.memory.sample_into_staging(
            self.batch_size
        )

        self._t_s.copy_(torch.from_numpy(bs))
        self._t_ns.copy_(torch.from_numpy(bns))
        non_blocking = self.device.type == "cuda"

        return (
            self.to_float_state(self._t_s.to(self.device, non_blocking=non_blocking)),
            self.to_float_state(self._t_ns.to(self.device, non_blocking=non_blocking)),
            torch.tensor(action, dtype=torch.long, device=self.device),
            torch.tensor(reward, dtype=torch.float32, device=self.device),
            torch.tensor(done, dtype=torch.bool, device=self.device),
        )


class MarioAgent(MarioRecall):
    def __init__(self, state_dim, action_dim, save_dir, device=None,
                 replay_capacity=100000):
        # Chain parent initializers to configure network, memory, and dimensions
        super().__init__(state_dim, action_dim, save_dir, device, replay_capacity)

        # Default matches the active gamma=0.99 lineage; train.py's
        # configure_agent() applies the configured value and stamps the
        # lineage guard. See config.py for why gamma is a LINEAGE
        # switch, not a casual knob.
        self.gamma = 0.99

        # Optimize only the online network. The target network is frozen
        # and synced manually, so it does not belong in the optimizer.
        self.optimizer = torch.optim.Adam(
            self.net.online_parameters, lr=0.00025
        )
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = int(1e5)      # min experiences before training
        self.learn_every = 3        # experiences between Q_online updates
        self.sync_every = int(1e4)  # experiences between Q_target syncs

    def td_estimate(self, state, action):
        """
        Get predicted Q-values for the selected action via online network.
        """
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        Double DQN calculation: Online net selects the best action;
        Target net evaluates its value.
        """
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)

        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        Compute loss and perform a backpropagation step.
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """
        Synchronize the target network weights with the online network.
        """
        self.net.sync_target()

    def buffer_ready(self):
        """True once the replay buffer holds enough experience to learn.

        Measured against the BUFFER, never curr_step: a resumed run
        restores curr_step from the checkpoint but starts with an
        EMPTY buffer, and a step-based gate would sample from it and
        crash on the very first learn step ("Sample larger than
        population"). Capped at maxlen so a small REPLAY_CAPACITY
        can't deadlock learning; floored at batch_size so sampling is
        always valid.
        """
        capacity = self.memory.maxlen or self.burnin
        return len(self.memory) >= max(min(self.burnin, capacity), self.batch_size)

    def learn_once(self):
        """One gradient update, UNGATED -- the vectorized caller owns
        burn-in (buffer_ready), the replay ratio (gradient debt), and
        target syncing (step delta). With N actors `curr_step % k` is
        unreliable: curr_step jumps by N and can step clean over the
        multiple, silently skipping target syncs forever."""
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    def save(self):
        """
        Save training checkpoint.

        Weights are moved to CPU before writing so a checkpoint trained
        on an AWS GPU loads cleanly on a CPU-only laptop (and vice
        versa) -- the file is device-agnostic.
        """
        if not self.save_dir:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"

        cpu_state = {k: v.cpu() for k, v in self.net.state_dict().items()}
        torch.save(
            {
                "model": cpu_state,
                "exploration_rate": self.exploration_rate,
                "curr_step": self.curr_step,
                # Adam moments -- lets a resume continue optimizing
                # exactly where it left off instead of restarting the
                # optimizer cold (which causes a brief loss spike).
                "optimizer": self.optimizer.state_dict(),
            },
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, checkpoint_path):
        """
        Restore a training checkpoint (model weights, exploration rate,
        and step counter) so a run can resume where it left off.

        map_location=self.device means a checkpoint written anywhere
        (local CPU run, GPU spot instance) loads onto whatever device
        this agent is using now.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Schema-tolerant restore. The extension (.chkpt/.pth/.pt) is
        # irrelevant -- torch.load sniffs the bytes -- but dict keys
        # vary across checkpoint generations:
        #   * this repo + Colab: {"model": ..., "exploration_rate",
        #     "curr_step"[, "optimizer"][, "best_eval"][, "episode"]}
        #   * bare-state_dict or "state_dict"-keyed files from early
        #     experiments
        # Missing keys keep the agent's current values, so a Colab
        # best-checkpoint loads clean.
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # bare state_dict file

        # ACTION-SET LINEAGE GUARD. The output layer's shape encodes
        # the action count; loading a 2-action checkpoint into this
        # 7-action net (or vice versa) fails deep inside
        # load_state_dict with a cryptic size mismatch. Check first
        # and fail with the actual explanation.
        ckpt_actions = checkpoint_action_dim(state_dict)
        if ckpt_actions is not None and ckpt_actions != self.action_dim:
            raise RuntimeError(
                f"ACTION-SET LINEAGE MISMATCH: {checkpoint_path} was trained "
                f"with {ckpt_actions} actions, but this agent has "
                f"{self.action_dim} (SIMPLE_MOVEMENT). Checkpoints are not "
                f"transferable across action sets -- the output layer shape "
                f"differs. Use a checkpoint from the matching lineage "
                f"(files carry '7_action' in their names)."
            )

        self.net.load_state_dict(state_dict)
        if isinstance(checkpoint, dict):
            self.exploration_rate = checkpoint.get(
                "exploration_rate", self.exploration_rate
            )
            self.curr_step = checkpoint.get("curr_step", self.curr_step)

            # Optimizer state (Adam moments) -- written by checkpoints
            # from the optimizer-resume fix onward. Older files simply
            # lack the key and start with fresh moments; a malformed
            # state (e.g. saved before a batch-size architecture change)
            # is skipped with a warning rather than killing the resume.
            opt_state = checkpoint.get("optimizer")
            if opt_state is not None and hasattr(self, "optimizer"):
                try:
                    self.optimizer.load_state_dict(opt_state)
                    print("[load] optimizer state restored")
                except Exception as exc:
                    print(f"[load] optimizer state skipped ({exc})")

        print(
            f"MarioNet loaded from {checkpoint_path} "
            f"(step {self.curr_step}, epsilon {self.exploration_rate:.4f})"
        )

    def learn(self):
        """
        Serial-loop learn(): periodically synchronize models,
        checkpoint, and optimize weights. The PARALLEL loop in train.py
        does NOT call this -- with N actors curr_step advances by N and
        the modulo gates below can skip clean over their multiples.
        It uses buffer_ready() + learn_once() and owns the target-sync
        cadence itself.
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        # Wait until enough experiences are cached -- measured against the
        # replay BUFFER, not curr_step. A resumed run restores curr_step
        # from the checkpoint but starts with an EMPTY buffer; a step-based
        # check would sample from a near-empty buffer and crash.
        if not self.buffer_ready():
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory (batch arrives on self.device)
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss
