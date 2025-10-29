from json import load

class Config:
    def __init__(self, path: str):
        with open(path, "r") as f:
            cfg = load(f)
        self.config = cfg

config = Config("config.json").config