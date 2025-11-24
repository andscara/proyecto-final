from dataclasses import dataclass

@dataclass
class ProfilerID:
    id: str

    def __str__(self):
        return self.id