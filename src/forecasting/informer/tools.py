from typing import Any
import numpy as np
import torch

def adjust_learning_rate(
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        learning_rate: float,
        lradj: str
    ):
    lr_adjust: dict[int, float] = {}
    if lradj=='type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        raise ValueError("Unsupported learning rate adjustment type {}".format(lradj))
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(
            self, 
            patience: int = 7, 
            verbose: bool = False, 
            delta: float = 0
        ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(
            self, 
            val_loss: np.floating[Any],
            model: torch.nn.Module
        ):
        path = ".informer_checkpoints/best_model_patience"
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
            val_loss: np.floating[Any], 
            model: torch.nn.Module, 
            path: str
        ):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth') #type: ignore
        self.val_loss_min = val_loss

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data: np.ndarray) -> np.ndarray: 
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean #type: ignore
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std #type: ignore
        return (data - mean) / std #type: ignore

    def inverse_transform(self, data: np.ndarray) -> np.ndarray: #type: ignore
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean #type: ignore
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std #type: ignore
        if data.shape[-1] != mean.shape[-1]: #type: ignore
            mean = mean[-1:] #type: ignore
            std = std[-1:] #type: ignore
        return (data * std) + mean #type: ignore