from dataclasses import dataclass
from . import Profiler, ProfilerID, SegmentationResult
import pandas as pd
from sklearn.cluster import KMeans

@dataclass(kw_only=True)
class KMeansProfilerID(ProfilerID):
    id: str = "kmeans_profiler"
    n_clusters: int

class KMeanProfiler(Profiler[KMeansProfilerID]):

    def __init__(self, id: KMeansProfilerID):
        super().__init__(id)
        assert id.n_clusters > 0, "Number of clusters must be positive."
        self._kmeans = None

    def profile(self, data: pd.DataFrame) -> SegmentationResult:
        # Initialize and fit KMeans model
        self._kmeans = KMeans(
            n_clusters=self.id.n_clusters
        )

        # Fit the model and get cluster labels
        cluster_labels = self._kmeans.fit_predict(data.values)

        # Create mapping from data index to cluster label
        mapping = {str(idx): str(label) for idx, label in enumerate(cluster_labels)}

        return SegmentationResult(
            id=self.id,
            mapping=mapping,
            average_data=self._calculate_average_data(data, mapping)
        )

    def clear(self) -> None:
        """Clear the stored KMeans model."""
        if self._kmeans is not None:
            del self._kmeans
            self._kmeans = None
