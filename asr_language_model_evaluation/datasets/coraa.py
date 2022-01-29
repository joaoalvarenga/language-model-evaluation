import pandas as pd
from asr_language_model_evaluation.datasets import Dataset


def _load_coraa(path: str):
    for chunk in pd.read_csv(path, chunksize=100):
        for i, row in chunk.iterrows():
            yield i, row['text']

class Coraa(Dataset):
    def __init__(self, path: str):
        super().__init__(path, _load_coraa(path))

    def get_name(self):
        return 'CORAA'