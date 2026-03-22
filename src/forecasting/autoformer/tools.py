import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import TypeVar, Generic, Iterator, overload
from pathlib import Path

plt.switch_backend('agg')


def adjust_learning_rate(
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        lradj: str = 'fixed',
        learning_rate: float = 0.0001
    ):
    """

    :type optimizer: torch.optim.Optimizer
    :param epoch: current epoch
    :param lradj: adjust learning rate [type1, type2]
    """

    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if lradj == 'fixed':
        return
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            4: 5e-5, 6: 1e-5, 8: 5e-6, 10: 1e-6,
            20: 5e-7, 17: 1e-7, 22: 5e-8
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(
        self, 
        patience: int =7,
        verbose: bool = False,
        delta: float = 0
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(
            self, 
            val_loss: float, 
            model: torch.nn.Module, 
            path: str
    ):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(
        self, 
        val_loss: float, 
        model: torch.nn.Module, 
        path: Path
    ):
        if self.verbose:
            print(f'\033[92mValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\033[0m')
        torch.save(model.state_dict(), path / 'checkpoint.pth')
        self.val_loss_min = val_loss


K = TypeVar("K", bound=str)
V = TypeVar("V")

class DotDict(dict[str, V], Generic[V]):
    """
    Diccionario con acceso por notación de punto:
    d.key <=> d["key"]
    """

    def __getattr__(self, key: str) -> V:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: V) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(key) from e


class StandardScaler():
    def __init__(
            self,
            mean: float, 
            std: float
        ):
        self.mean = mean
        self.std = std

    def transform(
            self, 
            data: NDArray[np.float32]
        ) -> NDArray[np.float32]:
        return (data - self.mean) / self.std

    def inverse_transform(
            self, 
            data: NDArray[np.float32]
        ) -> NDArray[np.float32]:
        return (data * self.std) + self.mean


# def visual(true, preds=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     plt.plot(true, label='GroundTruth', linewidth=2)
#     if preds is not None:
#         plt.plot(preds, label='Prediction', linewidth=2)
#     plt.legend()
#     plt.savefig(name, bbox_inches='tight')
