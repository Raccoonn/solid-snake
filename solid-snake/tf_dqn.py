from __future__ import annotations

import os
import random
import typing as t
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Tensor = torch.Tensor


@dataclass(frozen=True)
class DQNConfig:
    """
    DESCRIPTION: Hyperparameters/config for the DQN agent.

    PARAMETERS: alpha (REQ, float) - Learning rate.
                gamma (REQ, float) - Discount factor.
                epsilon (REQ, float) - Initial epsilon for epsilon-greedy policy.
                epsilon_dec (REQ, float) - Multiplicative epsilon decay.
                epsilon_end (REQ, float) - Minimum epsilon.
                batch_size (REQ, int) - Batch size for learning updates.
                input_dims (REQ, int) - Number of input features (state size).
                n_actions (REQ, int) - Number of discrete actions.
                fc1_dims (REQ, int) - First hidden layer width.
                fc2_dims (REQ, int) - Second hidden layer width.
                mem_size (REQ, int) - Replay buffer capacity.
                target_update_every (REQ, int) - Steps between target net updates.
                device (REQ, str) - Torch device string ('cpu', 'cuda', etc).
                fname (REQ, str) - Model checkpoint path.

    RETURNS: DQNConfig - Config object.
    """
    alpha: float
    gamma: float
    epsilon: float
    epsilon_dec: float
    epsilon_end: float
    batch_size: int
    input_dims: int
    n_actions: int
    fc1_dims: int = 512
    fc2_dims: int = 256
    mem_size: int = 100_000
    target_update_every: int = 1_000
    device: str = "cpu"
    fname: str = "snake_dqn.pt"


class ReplayBuffer:
    def __init__(self, max_size: int, input_dims: int) -> None:
        """
        DESCRIPTION: Fixed-size replay buffer storing transitions (s, a, r, s', done).

        PARAMETERS: max_size (REQ, int) - Maximum transitions stored.
                    input_dims (REQ, int) - State vector length.

        RETURNS: None
        """
        self.mem_size: int = int(max_size)
        self.mem_cntr: int = 0

        self.state_memory: np.ndarray = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory: np.ndarray = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory: np.ndarray = np.zeros((self.mem_size,), dtype=np.int64)
        self.reward_memory: np.ndarray = np.zeros((self.mem_size,), dtype=np.float32)
        self.done_memory: np.ndarray = np.zeros((self.mem_size,), dtype=np.float32)  # 1.0 if done else 0.0

    def store_transition(self, state: np.ndarray, action: int, reward: float, state_: np.ndarray, done: bool) -> None:
        """
        DESCRIPTION: Add a single transition to the replay buffer.

        PARAMETERS: state (REQ, np.ndarray) - Current state.
                    action (REQ, int) - Action taken.
                    reward (REQ, float) - Reward received.
                    state_ (REQ, np.ndarray) - Next state.
                    done (REQ, bool) - Episode terminal flag.

        RETURNS: None
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.astype(np.float32, copy=False)
        self.new_state_memory[index] = state_.astype(np.float32, copy=False)
        self.action_memory[index] = int(action)
        self.reward_memory[index] = float(reward)
        self.done_memory[index] = 1.0 if done else 0.0
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        DESCRIPTION: Sample a random minibatch from replay memory.

        PARAMETERS: batch_size (REQ, int) - Number of samples to draw.

        RETURNS: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] - (states, actions, rewards, next_states, dones)
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, int(batch_size), replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.done_memory[batch]
        return states, actions, rewards, states_, dones


class QNetwork(nn.Module):
    def __init__(self, input_dims: int, n_actions: int, fc1_dims: int, fc2_dims: int) -> None:
        """
        DESCRIPTION: Simple MLP Q-network.

        PARAMETERS: input_dims (REQ, int) - State size.
                    n_actions (REQ, int) - Number of actions.
                    fc1_dims (REQ, int) - Hidden layer 1 width.
                    fc2_dims (REQ, int) - Hidden layer 2 width.

        RETURNS: None
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dims),
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        DESCRIPTION: Forward pass.

        PARAMETERS: x (REQ, Tensor) - Input tensor [B, input_dims].

        RETURNS: Tensor - Q-values [B, n_actions].
        """
        return self.net(x)


class Agent:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_dec: float,
        epsilon_end: float,
        batch_size: int,
        input_dims: int,
        n_actions: int,
        fc1_dims: int = 512,
        fc2_dims: int = 256,
        mem_size: int = 100_000,
        fname: str = "dqn_model.pt",
        target_update_every: int = 1_000,
        device: str | None = None,
    ) -> None:
        """
        DESCRIPTION: DQN Agent (epsilon-greedy) with replay buffer and target network.

        PARAMETERS: alpha (REQ, float) - Learning rate.
                    gamma (REQ, float) - Discount factor.
                    epsilon (REQ, float) - Initial epsilon.
                    epsilon_dec (REQ, float) - Multiplicative epsilon decay.
                    epsilon_end (REQ, float) - Minimum epsilon.
                    batch_size (REQ, int) - Minibatch size.
                    input_dims (REQ, int) - State vector length.
                    n_actions (REQ, int) - Number of discrete actions.
                    fc1_dims (OPT, int), by default 512 - Hidden layer 1 width.
                    fc2_dims (OPT, int), by default 256 - Hidden layer 2 width.
                    mem_size (OPT, int), by default 100_000 - Replay buffer size.
                    fname (OPT, str), by default "dqn_model.pt" - Checkpoint file.
                    target_update_every (OPT, int), by default 1000 - Target update frequency (learn steps).
                    device (OPT, str|None), by default None - Torch device. If None -> CPU.

        RETURNS: None
        """
        self.action_space: list[int] = list(range(int(n_actions)))
        self.gamma: float = float(gamma)

        self.epsilon: float = float(epsilon)
        self.epsilon_dec: float = float(epsilon_dec)
        self.epsilon_min: float = float(epsilon_end)

        self.batch_size: int = int(batch_size)
        self.model_file: str = str(fname)

        self.learn_step_counter: int = 0
        self.target_update_every: int = int(target_update_every)

        self.device: torch.device = torch.device(device or "cpu")

        self.memory = ReplayBuffer(int(mem_size), int(input_dims))

        self.q_eval = QNetwork(int(input_dims), int(n_actions), int(fc1_dims), int(fc2_dims)).to(self.device)
        self.q_target = QNetwork(int(input_dims), int(n_actions), int(fc1_dims), int(fc2_dims)).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=float(alpha))
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss tends to be stable for DQN

    def remember(self, state: np.ndarray, action: int, reward: float, new_state: np.ndarray, done: int) -> None:
        """
        DESCRIPTION: Store a transition in replay memory.

        PARAMETERS: state (REQ, np.ndarray) - Current state.
                    action (REQ, int) - Action taken.
                    reward (REQ, float) - Reward received.
                    new_state (REQ, np.ndarray) - Next state.
                    done (REQ, int) - Terminal flag (1 if done else 0) to match your original call site.

        RETURNS: None
        """
        self.memory.store_transition(state, int(action), float(reward), new_state, bool(done))

    def choose_action(self, state: np.ndarray) -> int:
        """
        DESCRIPTION: Epsilon-greedy action selection.

        PARAMETERS: state (REQ, np.ndarray) - Current state vector.

        RETURNS: int - Action index.
        """
        # if random.random() < self.epsilon:
        #     return int(random.choice(self.action_space))

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
        with torch.no_grad():
            q_values = self.q_eval(state_t)  # [1, A]
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def learn(self) -> t.Optional[float]:
        """
        DESCRIPTION: Sample from replay buffer and perform a single DQN update step.

        PARAMETERS: None

        RETURNS: Optional[float] - Loss value if an update happened, else None.

        EXCEPTIONS: RuntimeError - If tensor ops fail unexpectedly.
        """
        if self.memory.mem_cntr < self.batch_size:
            return None

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)  # [B, 1]
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B, 1]
        next_states_t = torch.tensor(states_, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B, 1]

        # Q(s,a)
        q_pred = self.q_eval(states_t).gather(1, actions_t)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.q_target(next_states_t).max(dim=1, keepdim=True).values
            # If done, bootstrap term should be 0
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * q_next

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step_counter += 1

        # Periodically update target network
        if self.learn_step_counter % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        return float(loss.item())

    def save_model(self) -> None:
        """
        DESCRIPTION: Save the Q-network checkpoint to disk.

        PARAMETERS: None

        RETURNS: None

        EXCEPTIONS: OSError - If the file cannot be written.
        """
        os.makedirs(os.path.dirname(self.model_file) or ".", exist_ok=True)
        payload = {
            "q_eval": self.q_eval.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
        }
        torch.save(payload, self.model_file)

    def load_model(self, filename: str) -> None:
        """
        DESCRIPTION: Load the Q-network checkpoint from disk.

        PARAMETERS: filename (REQ, str) - Path to checkpoint file.

        RETURNS: None

        EXCEPTIONS: FileNotFoundError - If the checkpoint does not exist.
                    RuntimeError - If checkpoint tensors are incompatible.
        """
        payload = torch.load(filename, map_location=self.device)
        self.q_eval.load_state_dict(payload["q_eval"])
        self.q_target.load_state_dict(payload.get("q_target", payload["q_eval"]))
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.learn_step_counter = int(payload.get("learn_step_counter", self.learn_step_counter))
