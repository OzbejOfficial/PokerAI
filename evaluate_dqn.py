import torch
from rl_poker_env import RLPokerEnv
from dqn_agent import DQNAgent
from train_dqn import policy_to_action

def evaluate(model_path, episodes=100):
    env = RLPokerEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = 4  # fold, check, call, raise

    agent = DQNAgent(input_dim, output_dim)
    agent.load(model_path)

    wins = 0
    total_reward = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        action_seq = [policy_to_action(agent.act(obs, epsilon=0)) for _ in range(3)]
        obs, reward, done, _ = env.step(action_seq)

        if reward > 0:
            wins += 1
        total_reward += reward

        print(f"Episode {ep}: Reward={reward}, Stack={env.agent.stack}")

    print(f"\nResults over {episodes} episodes:")
    print(f"Win rate: {wins / episodes * 100:.1f}%")
    print(f"Average reward: {total_reward / episodes:.2f}")

if __name__ == "__main__":
    evaluate("checkpoints/dqn_ep200.pt", episodes=100)
