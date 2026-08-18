import json
import matplotlib.pyplot as plt


log_files = {

    "lr=0.0005": "logs/batch64_lr0_0005.json",

    "lr=0.001": "logs/batch64_lr0_001.json",

    "lr=0.002": "logs/batch64_lr0_002.json",
    "lr=0.003": "logs/batch64_lr0_003.json",
    "lr=0.004": "logs/batch64_lr0_004.json",
    "lr=0.004(1)": "logs/batch64_lr0_004(1).json",
}



plt.figure(figsize=(10, 5))


for label, path in log_files.items():

    with open(path, "r") as f:
        data = json.load(f)

    train_steps = []
    train_loss = []

    valid_steps = []
    valid_loss = []

    for item in data["history"]:
        if item["type"] == "train":
            train_steps.append(item["step"])
            train_loss.append(item["loss"])

        elif item["type"] == "valid":
            valid_steps.append(item["step"])
            valid_loss.append(item["loss"])

    # train curve
    plt.plot(
        train_steps,
        train_loss,
        label=f"{label} train"
    )

    # validation points
    plt.plot(
        valid_steps,
        valid_loss,
        marker="o",
        linestyle="--",
        label=f"{label} valid"
    )


plt.xlabel("Gradient Steps")
plt.ylabel("Loss")
plt.title("Learning Rate Sweep")
plt.ylim(1, 3)
plt.legend()
plt.grid()

plt.savefig("lr_sweep.png", dpi=300)

plt.show()