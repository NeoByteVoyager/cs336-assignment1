import json
import sys
from typing import cast

import torch
import numpy as np
from torch.cpu import device_count

from cs336_basics.transformer_lm import Model
from cs336_basics.crossentropy import crossEntropy
from cs336_basics.adamw import AdamW
from cs336_basics.gradient_clipping import gradientClipping
from cs336_basics.lr_cosine_schedule import lrCosineSchedule

from cs336_basics.data_loading import get_batch
from cs336_basics.checkpoint import load_checkpoint,save_checkpoint
from cs336_basics.logger import Logger

def train(model,
          config,
          optimizer,
          train_data_config,
          it: int,
          device):
    model.train()
    train_config = config["train"]
    lr = lrCosineSchedule(
        it,
        config["optimizer"]["lr"],
        config["scheduler"]["min_lr"],
        config["scheduler"]["warmup_steps"],
        train_config["iteration"]
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad()

    inputs, targets = get_batch(**train_data_config)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = crossEntropy(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

    loss.backward()

    gradientClipping(model.parameters(), train_config["gradient_clip"])

    optimizer.step()

    return loss.item(), lr

def valid(model, valid_data_config, device):
    model.eval()

    total_loss = 0.0
    eval_steps = 10
    with torch.no_grad():
        for _ in range(eval_steps):
            inputs, targets = get_batch(**valid_data_config)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = crossEntropy(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / eval_steps

def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Model(**config["model"]).to(device)

    # optimizer
    optimizer_config = {
        "params": model.parameters(),
        "lr": config["optimizer"]["lr"],
        "weight_decay": config["optimizer"]["weight_decay"],
        "betas": tuple(config["optimizer"]["betas"]),
        "eps": config["optimizer"]["eps"]
    }
    optimizer = AdamW(**optimizer_config)

    # train_data
    train_dataset = np.load("data/train_dataset.npy", mmap_mode='r')
    train_data_config = {
        "dataset": train_dataset,
        "batch_size": config["train"]["batch_size"],
        "context_length": config["train"]["context_length"],
        "device": device
    }
    # valid_data
    valid_data = np.load("data/val_dataset.npy", mmap_mode='r')
    valid_data_config = {
        "dataset": valid_data,
        "batch_size": config["train"]["batch_size"],
        "context_length": config["train"]["context_length"],
        "device": device
    }

    train_config = config["train"]
    for name, param in model.named_parameters():
        print(name, param.device)


    logger = Logger(config)
    # loop
    for it in range(train_config["iteration"]):
        loss, lr = train(model, config, optimizer,  train_data_config, it, device)
        logger.log_train(it + 1, loss, lr)
        if(it + 1) % 20 == 0:
            print(f"it:{it + 1}, train loss:{loss:.4f}")
        if (it + 1) % 100 == 0:
            loss = valid(model, valid_data_config, device)
            print(f"it:{it + 1}, valid loss:{loss:.4f}")
            logger.log_valid(it + 1, loss)
        if (it + 1) % train_config["save_interval"] == 0:
            save_checkpoint(model, optimizer, it + 1, train_config["checkpoint_path"])

    logger.save(
        config["train"]["log_path"]
    )

if __name__ == "__main__":
    main()