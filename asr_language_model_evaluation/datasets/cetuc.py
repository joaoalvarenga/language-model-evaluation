import glob
import os

from asr_language_model_evaluation.datasets import Dataset


def _load_cetuc(path: str):
    folders = glob.glob(os.path.join(path, '*'))
    i = 0
    for folder in folders:
        for filename in glob.glob(os.path.join(folder, '*.txt')):
            with open(filename) as f:
                text = f.read()
            yield i, text
            i += 1


class CETUC(Dataset):
    def __init__(self, path: str):
        super().__init__(path, _load_cetuc(path))

    def get_name(self):
        return 'CETUC'
