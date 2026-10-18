"""CPSC 4420 Assignment 2 - Q-learning on Taxi-v4.

Running this file performs every required undergraduate experiment, saves the
plots and numerical results, evaluates the best policy, and records an MP4.
"""

from pathlib import Path
import csv
import json

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("results")
EPISODES = 15000
MAX_STEPS = 200
SEED = 42

BASELINE = {
    "alpha": 0.5,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.9995,
    "epsilon_min": 0.05,
}


def moving_average(values, window=200):
    """Return a simple moving average for cleaner plots."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def choose_action(q_table, state, epsilon, rng):
    """Choose a random action with probability epsilon; otherwise be greedy."""
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))

    # Randomly choose among tied best actions so action 0 is not always favored.
    best_actions = np.flatnonzero(q_table[state] == np.max(q_table[state]))
    return int(rng.choice(best_actions))


def apply_action_noise(action, number_of_actions, noise_probability, rng):
    """With probability p, replace an action with a different valid action."""
    if noise_probability <= 0 or rng.random() >= noise_probability:
        return action

    other_actions = [candidate for candidate in range(number_of_actions)
                     if candidate != action]
    return int(rng.choice(other_actions))


def train_agent(config, episodes=EPISODES, seed=SEED, noise_probability=0.0):
    """Train one tabular Q-learning agent and return its Q-table and rewards."""
    env = gym.make("Taxi-v4", max_episode_steps=MAX_STEPS)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(seed)
    epsilon = config["epsilon_start"]
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0

        for _ in range(MAX_STEPS):
            intended_action = choose_action(q_table, state, epsilon, rng)
            action = apply_action_noise(
                intended_action,
                env.action_space.n,
                noise_probability,
                rng,
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Terminal states have no future Q-value.
            future_q = 0.0 if done else np.max(q_table[next_state])
            target = reward + config["gamma"] * future_q
            q_table[state, action] += config["alpha"] * (
                target - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        epsilon = max(config["epsilon_min"], epsilon * config["epsilon_decay"])

    env.close()
    return q_table, rewards


def evaluate_agent(q_table, episodes=100, seed=10000, noise_probability=0.0):
    """Evaluate greedily with exploration disabled."""
    env = gym.make("Taxi-v4", max_episode_steps=MAX_STEPS)
    rewards = []
    steps_list = []
    successes = 0
    rng = np.random.default_rng(seed)

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0

        for step in range(1, MAX_STEPS + 1):
            intended_action = int(np.argmax(q_table[state]))
            action = apply_action_noise(
                intended_action,
                env.action_space.n,
                noise_probability,
                rng,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                if terminated:
                    successes += 1
                rewards.append(total_reward)
                steps_list.append(step)
                break

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(successes / episodes),
        "mean_steps": float(np.mean(steps_list)),
        "episodes": episodes,
    }


def plot_baseline(rewards):
    """Save the baseline reward-versus-episode plot."""
    plt.figure(figsize=(10, 5.5))
    plt.plot(rewards, color="#9ecae1", alpha=0.25, linewidth=0.6,
             label="Episode reward")
    smooth = moving_average(rewards)
    x = np.arange(len(smooth)) + 199
    plt.plot(x, smooth, color="#08519c", linewidth=2,
             label="200-episode moving average")
    plt.xlabel("Training Episode")
    plt.ylabel("Total Reward")
    plt.title("Taxi-v4 Baseline Q-learning Training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_training.png", dpi=200)
    plt.close()


def plot_comparison(runs, title, filename):
    """Plot moving-average reward curves for one experiment."""
    plt.figure(figsize=(10, 5.5))
    for label, rewards in runs.items():
        smooth = moving_average(rewards)
        x = np.arange(len(smooth)) + 199
        plt.plot(x, smooth, linewidth=1.7, label=label)
    plt.xlabel("Training Episode")
    plt.ylabel("200-Episode Average Reward")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_policy_robustness(trained_results, original_policy_results):
    """Compare noise-trained agents with the original policy under noise."""
    labels = ["p = 0", "p = 0.10", "p = 0.25"]
    trained_rewards = [result["mean_reward"] for result in trained_results]
    original_rewards = [result["mean_reward"]
                        for result in original_policy_results]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width / 2, trained_rewards, width,
            label="Policy trained at this noise level")
    plt.bar(x + width / 2, original_rewards, width,
            label="Original p = 0 policy")
    plt.xticks(x, labels)
    plt.xlabel("Action Noise Probability")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Policy Robustness Under Action Noise")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "policy_robustness.png", dpi=200)
    plt.close()


def save_stochastic_summary(rows):
    """Save the stochastic-environment evaluation results."""
    fields = [
        "noise_probability", "evaluation_type", "training_episodes",
        "evaluation_episodes", "mean_reward", "success_rate", "mean_steps",
    ]
    with open(OUTPUT_DIR / "stochastic_summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(OUTPUT_DIR / "stochastic_summary.json", "w") as file:
        json.dump(rows, file, indent=2)


def record_video(q_table, filename="best_agent.mp4", seed=2026):
    """Record one greedy episode as an MP4."""
    env = gym.make("Taxi-v4", render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    state, _ = env.reset(seed=seed)
    frames = [env.render()]

    for _ in range(MAX_STEPS):
        action = int(np.argmax(q_table[state]))
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(OUTPUT_DIR / filename, frames, fps=3, codec="libx264")


def save_summary(rows):
    """Save experimental settings and evaluation metrics to CSV."""
    fields = [
        "experiment", "setting", "alpha", "gamma", "epsilon_start",
        "epsilon_decay", "training_episodes", "evaluation_episodes",
        "mean_reward", "success_rate", "mean_steps",
    ]
    with open(OUTPUT_DIR / "experiment_summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_configuration(experiment, setting, config, all_agents, rows):
    """Train, evaluate, and store one configuration."""
    print(f"Training {experiment}: {setting}")
    q_table, rewards = train_agent(config)
    metrics = evaluate_agent(q_table)
    rows.append({
        "experiment": experiment,
        "setting": setting,
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "epsilon_start": config["epsilon_start"],
        "epsilon_decay": config["epsilon_decay"],
        "training_episodes": EPISODES,
        "evaluation_episodes": metrics["episodes"],
        "mean_reward": metrics["mean_reward"],
        "success_rate": metrics["success_rate"],
        "mean_steps": metrics["mean_steps"],
    })
    all_agents[f"{experiment}: {setting}"] = (q_table, rewards, metrics, config)
    print(
        f"  reward={metrics['mean_reward']:.2f}, "
        f"success={100 * metrics['success_rate']:.1f}%, "
        f"steps={metrics['mean_steps']:.2f}"
    )
    return q_table, rewards, metrics


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_agents = {}
    rows = []

    baseline_q, baseline_rewards, baseline_metrics = run_configuration(
        "Baseline", "baseline", BASELINE.copy(), all_agents, rows
    )
    plot_baseline(baseline_rewards)

    alpha_runs = {}
    for alpha in [0.1, 0.5, 0.9]:
        config = BASELINE.copy()
        config["alpha"] = alpha
        _, rewards, _ = run_configuration(
            "Learning rate", f"alpha={alpha}", config, all_agents, rows
        )
        alpha_runs[f"alpha = {alpha}"] = rewards
    plot_comparison(
        alpha_runs,
        "Effect of Learning Rate on Taxi-v4 Q-learning",
        "learning_rate_comparison.png",
    )

    gamma_runs = {}
    for gamma in [0.50, 0.90, 0.99]:
        config = BASELINE.copy()
        config["gamma"] = gamma
        _, rewards, _ = run_configuration(
            "Discount factor", f"gamma={gamma}", config, all_agents, rows
        )
        gamma_runs[f"gamma = {gamma}"] = rewards
    plot_comparison(
        gamma_runs,
        "Effect of Discount Factor on Taxi-v4 Q-learning",
        "discount_factor_comparison.png",
    )

    epsilon_runs = {}
    for epsilon in [0.2, 0.6, 1.0]:
        config = BASELINE.copy()
        config["epsilon_start"] = epsilon
        _, rewards, _ = run_configuration(
            "Starting epsilon", f"epsilon={epsilon}", config, all_agents, rows
        )
        epsilon_runs[f"starting epsilon = {epsilon}"] = rewards
    plot_comparison(
        epsilon_runs,
        "Effect of Starting Exploration Rate on Taxi-v4 Q-learning",
        "epsilon_comparison.png",
    )

    # Extra-credit stochastic environment experiment. The selected action is
    # replaced by a different action with probability p.
    noise_levels = [0.0, 0.10, 0.25]
    noise_runs = {"p = 0": baseline_rewards}
    noise_trained_results = [baseline_metrics]
    original_policy_results = [baseline_metrics]
    stochastic_rows = [{
        "noise_probability": 0.0,
        "evaluation_type": "trained and evaluated at matching noise",
        "training_episodes": EPISODES,
        "evaluation_episodes": baseline_metrics["episodes"],
        "mean_reward": baseline_metrics["mean_reward"],
        "success_rate": baseline_metrics["success_rate"],
        "mean_steps": baseline_metrics["mean_steps"],
    }, {
        "noise_probability": 0.0,
        "evaluation_type": "original p=0 policy evaluated under noise",
        "training_episodes": EPISODES,
        "evaluation_episodes": baseline_metrics["episodes"],
        "mean_reward": baseline_metrics["mean_reward"],
        "success_rate": baseline_metrics["success_rate"],
        "mean_steps": baseline_metrics["mean_steps"],
    }]

    for noise_probability in noise_levels[1:]:
        print(f"Training Stochastic environment: p={noise_probability}")
        noisy_q, noisy_rewards = train_agent(
            BASELINE.copy(), noise_probability=noise_probability
        )
        matching_metrics = evaluate_agent(
            noisy_q, noise_probability=noise_probability
        )
        original_metrics = evaluate_agent(
            baseline_q, noise_probability=noise_probability
        )
        noise_runs[f"p = {noise_probability}"] = noisy_rewards
        noise_trained_results.append(matching_metrics)
        original_policy_results.append(original_metrics)

        for evaluation_type, metrics in [
            ("trained and evaluated at matching noise", matching_metrics),
            ("original p=0 policy evaluated under noise", original_metrics),
        ]:
            stochastic_rows.append({
                "noise_probability": noise_probability,
                "evaluation_type": evaluation_type,
                "training_episodes": EPISODES,
                "evaluation_episodes": metrics["episodes"],
                "mean_reward": metrics["mean_reward"],
                "success_rate": metrics["success_rate"],
                "mean_steps": metrics["mean_steps"],
            })

        print(
            f"  matching-noise reward={matching_metrics['mean_reward']:.2f}, "
            f"success={100 * matching_metrics['success_rate']:.1f}%"
        )
        print(
            f"  original-policy reward={original_metrics['mean_reward']:.2f}, "
            f"success={100 * original_metrics['success_rate']:.1f}%"
        )

    plot_comparison(
        noise_runs,
        "Effect of Action Noise on Taxi-v4 Q-learning",
        "stochastic_training_comparison.png",
    )
    plot_policy_robustness(noise_trained_results, original_policy_results)
    save_stochastic_summary(stochastic_rows)

    # Select by mean reward, then success rate, then fewer steps.
    best_name, best_data = max(
        all_agents.items(),
        key=lambda item: (
            item[1][2]["mean_reward"],
            item[1][2]["success_rate"],
            -item[1][2]["mean_steps"],
        ),
    )
    best_q, _, best_metrics, best_config = best_data
    np.save(OUTPUT_DIR / "best_q_table.npy", best_q)
    record_video(best_q)
    save_summary(rows)

    final_summary = {
        "baseline_configuration": BASELINE,
        "baseline_evaluation": baseline_metrics,
        "best_configuration_name": best_name,
        "best_configuration": best_config,
        "best_evaluation": best_metrics,
    }
    with open(OUTPUT_DIR / "final_summary.json", "w") as file:
        json.dump(final_summary, file, indent=2)

    print("\nFinished. Results are in the results folder.")
    print(f"Best configuration: {best_name}")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
