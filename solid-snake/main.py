from __future__ import annotations

import time
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch

from gui import get_run_config, RunConfig
from snakeEnvironment import snakeGame_v3
from tf_dqn import Agent


def save_progress_plot(avg_reward: list[float], epsilon_store: list[float], path: str = "progress.png") -> None:
    """
    DESCRIPTION: Save a simple training plot of average reward and epsilon.

    PARAMETERS: avg_reward (REQ, list[float]) - Average reward per episode (cumulative mean).
                epsilon_store (REQ, list[float]) - Epsilon values per episode.
                path (OPT, str), by default "progress.png" - Output image path.

    RETURNS: None
    """
    eps = list(range(1, len(avg_reward) + 1))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Avg Reward")
    ax1.plot(eps, avg_reward)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon")
    ax2.plot(eps, epsilon_store)

    plt.savefig(path)
    plt.close(fig)


def run_snake(config: RunConfig) -> int:
    """
    DESCRIPTION: Run the Snake DQN episode loop using the provided RunConfig.

    PARAMETERS: config (REQ, RunConfig) - Run configuration from GUI.

    RETURNS: int - Process exit code.
    """
    # Core environment params (can move into GUI later if you want)
    screen_width = 1200
    screen_height = 1200
    n_sqrs = 25
    input_dims = 12

    action_space = [0, 1, 2, 3]
    n_actions = len(action_space)

    # Agent params
    batch_size = 512
    fc1_dims = 512
    fc2_dims = 256

    agent = Agent(
        alpha=2.5e-4,
        gamma=0.95,
        epsilon=1.0,
        epsilon_dec=0.995,
        epsilon_end=0.01,
        batch_size=batch_size,
        input_dims=input_dims,
        n_actions=n_actions,
        mem_size=500_000,
        fc1_dims=fc1_dims,
        fc2_dims=fc2_dims,
        fname=config.checkpoint_path,
        target_update_every=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if config.load:
        try:
            agent.load_model(config.checkpoint_path)
            # For a demo, "load" usually means behave greedily
            if not config.train:
                agent.epsilon = 0.0
            print(f"\n\n... Model Loaded: {config.checkpoint_path} ...\n\n")
        except FileNotFoundError:
            print(f"\n\n... No checkpoint found at '{config.checkpoint_path}', starting fresh ...\n\n")

    env = snakeGame_v3(screen_width, screen_height, n_sqrs, difficulty=config.difficulty_fps)

    if config.show:
        env.setup_window()

    reward_store: list[float] = []
    avg_reward: list[float] = []
    epsilon_store: list[float] = []

    p_i, p_syms = 0, ("\\", "|", "/", "-")

    try:
        for episode in range(1, config.episodes + 1):
            total_reward = 0.0
            done, reward, state = env.reset()

            frame = 0
            start_time = time.time()
            print("\n")

            while not done:
                print("Playing a game...  " + p_syms[p_i], end="\r")
                p_i = (p_i + 1) % 4

                frame += 1

                if config.show:
                    env.render()

                action = agent.choose_action(state)
                done, reward, state_ = env.step(action, frame, buffer=config.buffer_steps)
                total_reward += float(reward)

                if config.train:
                    agent.remember(state, action, float(reward), state_, int(done))
                    agent.learn()

                state = state_

            reward_store.append(total_reward)
            avg_reward.append(float(np.mean(reward_store)))
            epsilon_store.append(float(agent.epsilon))

            save_progress_plot(avg_reward, epsilon_store, path="progress.png")

            print("\n\nEpisode: ", episode)
            print("Total reward: ", total_reward)
            print("Score:  ", env.score)
            print("Time elapsed: ", time.time() - start_time)

            # Save often for demos (and so you can stop anytime)
            try:
                agent.save_model()
            except Exception:
                pass

    finally:
        if config.show:
            env.shutdown()

    return 0


def main() -> int:
    """
    DESCRIPTION: Launch GUI to collect run configuration, then run Snake DQN.

    PARAMETERS: None

    RETURNS: int - Process exit code (0 for normal exit).
    """
    config = get_run_config()
    if config is None:
        print("Cancelled.")
        return 0
    return run_snake(config)


if __name__ == "__main__":
    raise SystemExit(main())
