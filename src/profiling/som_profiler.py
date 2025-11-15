import pandas as pd
from profiling.profiler import Profiler, ProfilerID
import torch
import torch.nn as nn
from dataclasses import dataclass   
from profiling.profiling_result import SegmentationResult

@dataclass
class SomProfilerID(ProfilerID):
    rows: int
    cols: int
    learning_rate: float
    sigma: float | None
    epochs: int
    id: str = "som_profiler"


class SomProfiler(Profiler, nn.Module):

    def __init__(
            self,
            rows: int = 10,
            cols: int = 10,
            learning_rate: float = 0.5,
            sigma: float | None = None,
            epochs: int = 10,
            batch_size: int = 32,
            device: str = "cpu"
        ):
        id = SomProfilerID(
            rows=rows,
            cols=cols,
            learning_rate=learning_rate,
            sigma=sigma if sigma is not None else max(rows, cols) / 2,
            epochs=epochs
        )
        super().__init__(id)
        assert rows > 0, "Number of rows must be positive."
        self._rows = rows
        assert cols > 0, "Number of columns must be positive."
        self._cols = cols
        assert 0 < learning_rate <= 1, "Learning rate must be in the range (0, 1]."
        self._lr0 = learning_rate
        if sigma is None:
            sigma = max(rows, cols) / 2
        self._sigma0 = sigma
        assert epochs > 0, "Number of epochs must be positive."
        self._epochs = epochs
        assert batch_size > 0, "Batch size must be positive."
        self._batch_size = batch_size
        assert device in ["cpu", "cuda", "mps"], "Device must be either 'cpu', 'cuda', or 'mps'"
        self._device = device

        # coordenadas de cada neurona (m*n, 2)
        coords = torch.stack(torch.meshgrid(
            torch.arange(self._rows), torch.arange(self._cols), indexing="ij"
        ), dim=2).reshape(-1, 2).float()

        self.register_buffer("coords", coords)


    def find_bmu(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, dim)
        # weights: (M, dim)
        dists = torch.cdist(batch, self._weights)  # (B, M)
        return torch.argmin(dists, dim=1)         # (B,)

    def neighborhood(self, bmu_indices: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Distancia entre todas las neuronas y cada BMU
        # coords: (M, 2), coords[bmu_indices]: (B, 2)
        dist = torch.cdist(self.coords, self.coords[bmu_indices])  # (M, B)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))           # (M, B)
    
    def partial_fit(
            self, 
            batch: torch.Tensor, 
            epoch: int, 
            total_epochs: int
        ) -> None:

        # decaimiento exponencial
        lr = self._lr0 * torch.exp(torch.tensor(-epoch / total_epochs))
        sigma = self._sigma0 * torch.exp(torch.tensor(-epoch / total_epochs))

        # BMUs
        bmu_idx = self.find_bmu(batch)

        # matriz de vecindad (M, B)
        h = self.neighborhood(bmu_idx, sigma).unsqueeze(-1)  # (M, B, 1)

        # batch expandido
        x = batch.unsqueeze(0)  # (1, B, dim)

        # weights: (M, dim) -> (M, 1, dim)
        w = self._weights.unsqueeze(1)

        # update vectorizado
        influence = h * (x - w)  # (M, B, dim)
        influence = influence.mean(dim=1)  # (M, dim)

        self._weights.data += lr * influence


    def profile(self, data: pd.DataFrame) -> SegmentationResult:
        self._weights = torch.randn((self._rows * self._cols, data.shape[1]), device=self._device)
        self._data = torch.tensor(data.values, dtype=torch.float32, device=self._device)
        N = self._data.size(0)

        for epoch in range(self._epochs):
            perm = torch.randperm(N)
            for i in range(0, N, self._batch_size):
                batch = self._data[perm[i:i+self._batch_size]]
                self.partial_fit(batch, epoch, self._epochs)

        # Asignar cada punto de datos a su BMU
        bmu_indices = self.find_bmu(self._data)  # (N,)
        mapping = {str(idx): str(bmu_idx.item()) for idx, bmu_idx in enumerate(bmu_indices)}
        return SegmentationResult(
            id=self._id, 
            mapping=mapping,
            average_data=self._calculate_average_data(data, mapping)
        )

    def clear(self) -> None:
        # Implementation to clear any stored state or data
        if hasattr(self, "_weights"):
            del self._weights
        if hasattr(self, "_data"):
            del self._data