import numpy as np

class rlBase:
    def __init__(self) -> None:
        pass

    def action(self, obs:np.ndarray):     # return action or (action, logp)
        raise NotImplementedError

    def train(self):
        raise NotImplementedError