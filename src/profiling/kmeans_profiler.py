from dataclasses import dataclass
from . import Profiler, ProfilerID, SegmentationResult
import pandas as pd

@dataclass(kw_only=True)
class KMeansProfilerID(ProfilerID):
    id: str = "kmeans_profiler"
    n_clusters: int

class KMeanProfiler(Profiler[KMeansProfilerID]):

    def __init__(self, id: KMeansProfilerID):
        super().__init__(id)
    
    def profile(self, data: pd.DataFrame) -> SegmentationResult:
        pass
