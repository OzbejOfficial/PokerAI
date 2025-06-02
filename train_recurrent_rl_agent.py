import os
import torch
from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from rl_poker_env import RLPokerEnv

# Create environment
env = DummyVecEnv([lambda: RLPokerEnv(num_opponents=1)])

# Define model
model = RecurrentPPO(
    "MultiInputLstmPolicy",  # Important for RecurrentPPO
    env,
    verbose=1,
    tensorboard_log="./tensorboard_rl_poker/",
    n_steps=64,              # Shorter episodes, smaller n_steps recommended
    batch_size=32,
    learning_rate=3e-4,
    gamma=0.99
)

# Training loop
total_timesteps = 50_000
model_path = "models/recurrent_ppo_poker"
os.makedirs("models", exist_ok=True)

# Train and periodically save
for i in range(0, total_timesteps, 10_000):
    model.learn(total_timesteps=10_000, reset_num_timesteps=False)
    model.save(f"{model_path}_step_{i+10_000}")
    print(f"✅ Saved model at step {i+10_000}")

print("🏁 Training complete.")
env.close()
