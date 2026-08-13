import json
import matplotlib.pyplot as plt


with open("logs/baseline.json", "r") as f:
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


plt.figure(figsize=(10, 5))

plt.plot(
    train_steps,
    train_loss,
    label="train loss"
)

plt.plot(
    valid_steps,
    valid_loss,
    marker="o",
    label="valid loss"
)


plt.xlabel("Gradient Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.legend()
plt.grid()

plt.savefig("loss_curve.png", dpi=300)

plt.show()