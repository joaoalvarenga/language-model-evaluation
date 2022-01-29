from abc import ABC, abstractmethod

from asr_language_model_evaluation.preprocessing import normalize
from typing import Iterator


class Dataset(ABC):
    def __init__(self, path: str, items: Iterator) -> None:
        self.items = items
        self.path = path

    def __iter__(self):
        return self

    def _next(self):
        return next(self.items)

    @abstractmethod
    def get_name(self):
        raise NotImplemented()

    def __next__(self):
        __id, text = self._next()
        return {'dataset': self.get_name(), 'file_line': __id, 'filename': self.path, 'text': text}
        #return normalize(text)