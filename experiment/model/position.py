import json
import matplotlib.pyplot as plt


log_files = {


    "rope": "logs/batch64_lr0_003_warmup200.json",
    "nope": "logs/nope_lr0_003_warm200.json",
    "it3000": "logs/it3000_lr0_003_warm200.json"
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
plt.title("Position")
plt.ylim(1, 2.5)
plt.legend()
plt.grid()

plt.savefig("position.png", dpi=300)

plt.show()