from typing import Callable
from abc import ABC, abstractmethod

class Sweeping(ABC):
    def __init__(self, funct: Callable = None):
        if funct is not None and not callable(funct):
            raise ValueError("Parameter 'funct' must be a lambda funct. or not defined.")
        self.funct = funct

    @abstractmethod
    def _apply(self):
        """Subclasses must implement this."""
        pass