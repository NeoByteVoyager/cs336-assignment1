import json
import matplotlib.pyplot as plt


log_files = {
    #"batch16": ("logs/batch16.json", 16),
    "batch32": ("logs/batch32.json", 32),
    "batch64": ("logs/batch64.json", 64),
    #"batch128": ("logs/baseline_tpu_2000.json", 128),
}

num_devices = 8
context_length = 256


plt.figure(figsize=(10, 5))


for label, (path, batch_size) in log_files.items():

    with open(path, "r") as f:
        data = json.load(f)

    train_tokens = []
    train_loss = []

    valid_tokens = []
    valid_loss = []

    for item in data["history"]:

        tokens = (
            item["step"]
            * batch_size
            * num_devices
            * context_length
        )

        if item["type"] == "train":
            train_tokens.append(tokens)
            train_loss.append(item["loss"])

        elif item["type"] == "valid":
            valid_tokens.append(tokens)
            valid_loss.append(item["loss"])

    plt.plot(
        train_tokens,
        train_loss,
        label=f"{label} train"
    )

    plt.plot(
        valid_tokens,
        valid_loss,
        marker="o",
        linestyle="--",
        label=f"{label} valid"
    )


plt.xlabel("Training Tokens")
plt.ylabel("Loss")
plt.title("Batch Size Experiment")
plt.ylim(1, 3)
plt.legend()
plt.grid()

plt.savefig("batch_size_experiment.png", dpi=300)

plt.show()