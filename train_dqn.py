import numpy as np
import os
import csv
from rl_poker_env import RLPokerEnv
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

EPISODES = 10000
BATCH_SIZE = 64
REPLAY_CAPACITY = 5000
SAVE_EVERY = 200
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9995

def policy_to_action(index):
    return [
        {"wanted_action": "fold", "raise_amount": 0, "call_till": 0, "action_vs_raise": "fold", "reraise_amount": 0},
        {"wanted_action": "check", "raise_amount": 0, "call_till": 0, "action_vs_raise": "fold", "reraise_amount": 0},
        {"wanted_action": "call", "raise_amount": 10, "call_till": 100, "action_vs_raise": "call", "reraise_amount": 20},
        {"wanted_action": "raise", "raise_amount": 30, "call_till": 30, "action_vs_raise": "reraise", "reraise_amount": 40}
    ][index]

def train():
    env = RLPokerEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = 4  # Discrete: fold, check, call, raise
    agent = DQNAgent(input_dim, output_dim)
    buffer = ReplayBuffer(REPLAY_CAPACITY)

    epsilon = EPSILON_START
    os.makedirs("checkpoints", exist_ok=True)

    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Final Stack", "Loss"])

        for ep in range(1, EPISODES + 1):
            obs = env.reset()
            action_seq = [policy_to_action(agent.act(obs, epsilon)) for _ in range(3)]
            next_obs, reward, done, _ = env.step(action_seq)
            # env.render()

            buffer.push(obs, agent.act(obs, epsilon), reward, next_obs, done)
            loss = 0

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = agent.train_step(batch)

            writer.writerow([ep, reward, env.agent.stack, loss])
            print(f"Ep {ep}: Reward={reward:.2f}, Stack={env.agent.stack}, Loss={loss:.4f}, Epsilon={epsilon:.2f}")

            if ep % SAVE_EVERY == 0:
                print(f"Saving model at episode {ep}")
                agent.save(f"checkpoints/dqn_ep{ep}.pt")

            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

if __name__ == "__main__":
    train()
