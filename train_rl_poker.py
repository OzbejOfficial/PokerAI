import numpy as np
import csv
import os
import pickle
from rl_poker_env import RLPokerEnv


def generate_action_policy(obs):
    strength = obs[-1]  # Hand strength (0 to 1)
    base_raise = 20

    if strength > 0.85:
        return {
            "wanted_action": "raise",
            "raise_amount": base_raise * 2,
            "call_till": 100,
            "action_vs_raise": "reraise",
            "reraise_amount": base_raise * 3
        }
    elif strength > 0.65:
        return {
            "wanted_action": "call",
            "raise_amount": base_raise,
            "call_till": 40,
            "action_vs_raise": "call",
            "reraise_amount": base_raise * 2
        }
    elif strength > 0.4:
        return {
            "wanted_action": "check",
            "raise_amount": 0,
            "call_till": 10,
            "action_vs_raise": "fold",
            "reraise_amount": 0
        }
    else:
        return {
            "wanted_action": "fold",
            "raise_amount": 0,
            "call_till": 0,
            "action_vs_raise": "fold",
            "reraise_amount": 0
        }


def train(num_episodes=1000, log_file="training_log.csv", save_every=50, resume=True):
    env = RLPokerEnv()
    os.makedirs("checkpoints", exist_ok=True)

    start_ep = 0
    if resume and os.path.exists("checkpoints/state.pkl"):
        with open("checkpoints/state.pkl", "rb") as f:
            start_ep = pickle.load(f).get("episode", 0)
        print(f"Resuming training from episode {start_ep + 1}")

    log_mode = "a" if resume and os.path.exists(log_file) else "w"
    with open(log_file, log_mode, newline="") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["Episode", "Reward", "Final Stack"])

        for episode in range(start_ep, num_episodes):
            obs = env.reset()
            actions = [generate_action_policy(obs) for _ in range(3)]
            obs, reward, done, _ = env.step(actions)
            env.render()

            writer.writerow([episode + 1, reward, env.agent.stack])
            print(f"Episode {episode + 1}: Reward={reward:.2f}, Stack={env.agent.stack}")

            # Save checkpoint
            if (episode + 1) % save_every == 0:
                with open("checkpoints/state.pkl", "wb") as f:
                    pickle.dump({"episode": episode + 1}, f)


if __name__ == "__main__":
    train(50000)
