from stable_baselines3 import PPO
from rl_poker_env import RLPokerEnv

env = RLPokerEnv(num_opponents=1)

# Load a trained model
model = PPO.load("models/ppo_poker.zip")

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    total_reward += reward

print(f"\n🏁 Final Episode Reward: {total_reward}")
