import logging
from profiling.profiler import Profiler
from profiling.profiling_result import Profile
import pandas as pd


class ProfilingManager:

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._profilers: list[Profiler] = []

    def run_profilers(self, data: pd.DataFrame) -> list[Profile]:
        results: list[Profile] = []
        for profiler in self._profilers:
            self._logger.info(f"Running profiler: {profiler.id}")
            profiling_result = profiler.profile(data)
            self._logger.info(f"Profiler {profiler.id} completed with {len(profiling_result.mapping)} profiles.")
            results.append(profiling_result)
            self._logger.info(f"Clearing profiler: {profiler.id}")
            profiler.clear()
        return results
    
    
