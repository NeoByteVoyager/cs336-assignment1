import torch
import numpy as np

from cs336_basics.transformer_lm import Model
from cs336_basics.crossentropy import crossEntropy
from cs336_basics.adamw import AdamW
from cs336_basics.gradient_clipping import gradientClipping
from cs336_basics.lr_cosine_schedule import lrCosineSchedule

from cs336_basics.data_loading import get_batch
from cs336_basics.checkpoint import load_checkpoint,save_checkpoint


def train(model,
          optimizer,
          train_config,
          train_data_config,
          it: int):
    model.train()

    lr = lrCosineSchedule(
        it,
        1e-3,
        1e-5,
        100,
        train_config["iteration"]
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad()

    inputs, targets = get_batch(**train_data_config)
    outputs = model(inputs)

    loss = crossEntropy(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
    loss.backward()

    gradientClipping(model.parameters(), 1.0)

    optimizer.step()

    if (it + 1) % 10 == 0:
        print(it, loss.item())

    if (it + 1) % train_config["save_interval"] == 0:
        save_checkpoint(model, optimizer, it, train_config["checkpoint_path"])


def valid(model, valid_data_config):
    model.eval()

    total_loss = 0.0
    eval_steps = 5
    with torch.no_grad():
        for _ in range(eval_steps):
            inputs, targets = get_batch(**valid_data_config)

            outputs = model(inputs)

            loss = crossEntropy(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            total_loss += loss.item()
    print(f"valid_loss: {total_loss / eval_steps}")
    model.train()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # model
    model_config = {
        "vocab_size": 10000,
        "context_length": 100,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 8,
        "d_ff": 1024,
        "rope_theta": 10000
    }
    model = Model(**model_config).to(device)

    # optimizer
    optimizer_config = {
        "params": model.parameters(),
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "betas": (0.9, 0.99),
        "eps": 1e-8
    }
    optimizer = AdamW(**optimizer_config)
    # train_data
    train_dataset = np.load("data/train_dataset.npy", mmap_mode='r')
    train_data_config = {
        "dataset": train_dataset,
        "batch_size": 32,
        "context_length": 100,
        "device": device
    }
    # valid_data
    valid_data = np.load("data/val_dataset.npy", mmap_mode='r')
    valid_data_config = {
        "dataset": valid_data,
        "batch_size": 32,
        "context_length": 100,
        "device": device
    }

    train_config = {
        "iteration": 1000,
        "save_interval": 100,
        "checkpoint_path": "ckpt.pt"
    }
    for name, param in model.named_parameters():
        print(name, param.device)
    # loop
    for it in range(train_config["iteration"]):
        train(model, optimizer,  train_config, train_data_config, it)

        if (it + 1) % 100 == 0:
            valid(model, valid_data_config)

if __name__ == "__main__":
    main()