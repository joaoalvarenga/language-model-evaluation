import pandas as pd
from asr_language_model_evaluation.datasets import Dataset


def _load_cv(path: str):
    for chunk in pd.read_csv(path, chunksize=100, sep='\t'):
        for i, row in chunk.iterrows():
            yield i, row['sentence']


class CommonVoice(Dataset):
    def __init__(self, path: str):
        super().__init__(path, _load_cv(path))

    def get_name(self):
        return 'CommonVoice'
