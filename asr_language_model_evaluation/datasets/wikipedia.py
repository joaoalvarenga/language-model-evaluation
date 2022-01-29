from asr_language_model_evaluation.datasets import Dataset


def _load_corpus(path: str):
    with open(path) as f:
        for __id, line in enumerate(f):
            text = line.strip()
            yield __id, text


class Wikipedia(Dataset):
    def get_name(self):
        return 'Wikipedia'

    def __init__(self, path: str):
        super().__init__(path, _load_corpus(path))
