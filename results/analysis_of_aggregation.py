import pandas as pd
import matplotlib.pyplot as plt

def plot_agg_csv(csv_path, title, save_path=None):
    """
    Plot aggregation times from a CSV file.
    """
    df = pd.read_csv(csv_path, index_col=0)

    # column names are round numbers (string), convert to int
    rounds = df.columns.astype(int)

    gpu = df.loc["GPU"].values
    cpu = df.loc["CPU"].values

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, gpu, marker="o", label="GPU")
    plt.plot(rounds, cpu, marker="o", label="CPU")
    plt.xlabel("Federated Round")
    plt.ylabel("Aggregation Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- save image here ---
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # -----------------------

    plt.show()
    plt.close()


# CIFAR-10 ResNet50
plot_agg_csv(
    "./CIFAR10_ResNet_aggregation_times.csv",
    "CIFAR-10 ResNet50 Aggregation Time per Round",
    save_path="./CIFAR10_ResNet_agg.png"
)

# LSTM + CYBER-THREAT
plot_agg_csv(
    "./LSTM_CYBER_aggregation_times.csv",
    "LSTM-CYBER Aggregation Time per Round",
    save_path="./LSTM_CYBER_agg.png"
)
