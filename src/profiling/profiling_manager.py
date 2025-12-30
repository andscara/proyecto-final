import logging
from . import Profiler
from . import SegmentationResult
import pandas as pd

from storage.storage import BaseStorage


class ProfilingManager:

    def __init__(
            self,
            logger: logging.Logger,
            storage: BaseStorage
        ):
        self._logger = logger
        self._storage = storage
        self._profilers: list[Profiler] = []

    def run_profilers(self, data: pd.DataFrame) -> list[SegmentationResult]:
        results: list[SegmentationResult] = []
        for profiler in self._profilers:
            existing_result = self._storage.get_segmentation_result(str(profiler.id))
            if existing_result is not None:
                self._logger.info(f"Skipping profiler {profiler.id} as result already exists in storage.")
                results.append(SegmentationResult(profiler.id, existing_result[0], existing_result[1]))
                continue

            self._logger.info(f"Running profiler: {profiler.id}")
            profiling_result = profiler.profile(data)
            self._logger.info(f"Profiler {profiler.id} completed with {len(profiling_result.mapping)} profiles.")
            results.append(profiling_result)
            self._logger.info(f"Clearing profiler: {profiler.id}")
            profiler.clear()
        return results
    
    
