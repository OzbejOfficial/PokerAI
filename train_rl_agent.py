from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rl_poker_env import RLPokerEnv
import os

env = make_vec_env(lambda: RLPokerEnv(num_opponents=1), n_envs=1)

model_path = "models/ppo_poker"
os.makedirs("models", exist_ok=True)

if os.path.exists(f"{model_path}.zip"):
    print("✅ Loading existing model...")
    model = PPO.load(model_path, env=env)
else:
    print("🆕 Creating new PPO model...")
    model = PPO("MultiInputPolicy", env, verbose=1)

total_timesteps = 100_000
save_interval = 10_000

for step in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    model.save(model_path)
    print(f"📦 Saved checkpoint at {step + save_interval} steps.")

print("🏁 Training complete.")
env.close()
