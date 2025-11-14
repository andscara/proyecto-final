from dataclasses import dataclass

from profiling.profiler import ProfilerID

@dataclass
class Profile:
    """
    Class representing the result of a profiling operation.
    Attributes:
        id (ProfilerID): Identifier of the profiler that generated this result.
        mapping (dict[str, str]): Dictionary containing a mapping between the smart meter ID (key) and its associated cluster number (value).
    """
    id: ProfilerID
    mapping: dict[str, str]