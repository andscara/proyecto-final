from profiling.profiler import Profiler


class ProfilingManager:

    def __init__(self):
        self.profilers: dict[str, Profiler] = {}

    def register_profiler(self, profiler: Profiler):
        self.profilers[profiler.get_name()] = profiler

    def run_profilers(self, data) -> dict[str, dict[str, str]]:
        results = {}
        for name, profiler in self.profilers.items():
            results[name] = profiler.profile(data)
        return results
    
    
