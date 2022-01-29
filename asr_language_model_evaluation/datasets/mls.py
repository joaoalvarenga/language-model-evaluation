import pandas as pd
from asr_language_model_evaluation.datasets import Dataset


def _load_mls(path: str):
    for chunk in pd.read_csv(path, chunksize=100, header=None, sep='\t'):
        for i, row in chunk.iterrows():
            yield i, row[1]


class MultilingualLibriSpeech(Dataset):
    def __init__(self, path: str):
        super().__init__(path, _load_mls(path))

    def get_name(self):
        return 'Multilingual LibriSpeech'
