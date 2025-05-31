import matplotlib.pyplot as plt
import pandas as pd
import os

log_path = "logs/poker_log.csv"
if not os.path.exists(log_path):
    print("No log file found.")
else:
    df = pd.read_csv(log_path)
    df["rolling_reward"] = df["stack_delta"].rolling(window=100).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df["episode"], df["stack_delta"], alpha=0.4, label="Reward per Episode")
    plt.plot(df["episode"], df["rolling_reward"], linewidth=2, label="Rolling Avg (100)", color="blue")
    plt.title("RL Agent Poker Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Δ Stack)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
