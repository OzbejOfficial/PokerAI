
import pandas as pd
import matplotlib.pyplot as plt
import os

log_path = "logs/poker_log.csv"
if not os.path.exists(log_path):
    print("CSV log file not found.")
    exit()

df = pd.read_csv(log_path)

# Map result to score
score_map = {"win": 1, "tie": 0, "loss": -1}
df["score"] = df["result"].map(score_map)
print(df.head())

# Compute cumulative sum
df["cumulative_score"] = df["score"].cumsum()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["cumulative_score"], marker='o', linestyle='-')
plt.title("RL Agent Performance Over Time")
plt.xlabel("Episode")
plt.ylabel("Cumulative Score (+1 win, 0 tie, -1 loss)")
plt.grid(True)
plt.tight_layout()
plt.show()
