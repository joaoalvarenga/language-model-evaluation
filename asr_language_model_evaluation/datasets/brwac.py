from asr_language_model_evaluation.datasets import Dataset
import re


def _load_brwac(data_dir: str):
    with open(data_dir, encoding="utf-8") as f:

        add_space = 1
        doc_id, title, uri = None, None, None
        current_sentence, current_paragraph_sentences, text = "", [], []
        id_ = 0
        for line in f:

            line = line.strip()

            if line not in ["<p>", "<s>"]:  # skip these tags

                if line.startswith("<doc"):  # doc begin
                    doc_id = re.findall('docid="(.*?)"', line)[0]
                    title = re.findall('title="(.*?)"', line)[0]
                    uri = re.findall('uri="(.*?)"', line)[0]

                elif line == "<g/>":  # don't add space with <g/> occurrence
                    add_space = 0

                elif line == "</s>":  # end sentence
                    current_paragraph_sentences.append(current_sentence)
                    current_sentence = ""

                elif line == "</p>":  # end paragraph
                    text.append({"paragraphs": current_paragraph_sentences})
                    current_paragraph_sentences = []

                elif len(current_sentence) == 0:
                    current_sentence = line

                else:
                    current_sentence = (add_space * " ").join([current_sentence, line])
                    add_space = 1

                if line.strip() == "</doc>":  # doc end
                    #yield id_, text
                    for paragraph in text:
                        for sentence in paragraph['paragraphs']:
                            yield id_, sentence
                            id_ += 1
                    #yield id_, {"doc_id": doc_id, "title": title, "uri": uri, "text": text}
                    #id_ += 1
                    add_space = 1
                    doc_id, title, uri = None, None, None
                    current_sentence, current_paragraph_sentences, text = "", [], []


class BRWaC(Dataset):
    def get_name(self):
        return 'BRWaC'

    def __init__(self, path: str):
        super().__init__(path, _load_brwac(path))

