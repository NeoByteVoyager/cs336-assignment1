import time
import json


class Logger:

    def __init__(self, config):
        self.config = config
        self.history = []
        self.start_time = time.time()


    def log_train(self, step, loss, lr=None):

        self.history.append({
            "type": "train",
            "step": step,
            "time": time.time() - self.start_time,
            "loss": loss,
            "lr": lr
        })


    def log_valid(self, step, loss):

        self.history.append({
            "type": "valid",
            "step": step,
            "time": time.time() - self.start_time,
            "loss": loss
        })


    def save(self, path):

        data = {
            "config": self.config,
            "history": self.history
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)