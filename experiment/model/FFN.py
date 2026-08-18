import json
import matplotlib.pyplot as plt


log_files = {

    "swiglu": "logs/batch64_lr0_003_warmup200.json",
    "silu": "logs/silu_lr0_003_warm200.json",

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
plt.title("FFN")
plt.ylim(1, 2.5)

plt.legend()
plt.grid()

plt.savefig("ffn.png", dpi=300)

plt.show()