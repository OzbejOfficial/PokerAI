import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_log(path="training_log.csv", smooth_window=20):
    df = pd.read_csv(path)

    df["Episode"] = df["Episode"].astype(int)
    df["Reward"] = pd.to_numeric(df["Reward"], errors="coerce").fillna(0)
    df["Final Stack"] = pd.to_numeric(df["Final Stack"], errors="coerce").fillna(0)

    # Compute cumulative reward
    df["Cumulative Reward"] = df["Reward"].cumsum()
    # Moving averages
    df["Smoothed Reward"] = df["Reward"].rolling(window=smooth_window, min_periods=1).mean()
    df["Smoothed Stack"] = df["Final Stack"].rolling(window=smooth_window, min_periods=1).mean()

    plt.figure(figsize=(12, 8))

    # Smoothed Reward
    plt.subplot(3, 1, 1)
    sns.lineplot(data=df, x="Episode", y="Smoothed Reward", label=f"{smooth_window}-ep Moving Avg")
    plt.title("Smoothed Reward")
    plt.ylabel("Reward")

    # Cumulative Reward
    plt.subplot(3, 1, 2)
    sns.lineplot(data=df, x="Episode", y="Cumulative Reward", color="green")
    plt.title("Cumulative Reward")
    plt.ylabel("Total Reward")

    # Smoothed Stack
    plt.subplot(3, 1, 3)
    sns.lineplot(data=df, x="Episode", y="Smoothed Stack", label=f"{smooth_window}-ep Moving Avg", color="orange")
    plt.title("Smoothed Final Stack")
    plt.ylabel("Stack")
    plt.xlabel("Episode")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_log()
