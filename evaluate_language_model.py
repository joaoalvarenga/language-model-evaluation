import argparse
import pandas as pd
import os

from typing import Set
from datasets import load_dataset, load_metric, Dataset
from transformers import Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

chars_to_ignore_regex = '[\,\?\.\!\;\:\"]'  # noqa: W605
wer_metric = load_metric("wer")


def clean_model_name(text: str) -> str:
    text = text.replace('.binary', '')
    text = text.replace('.arpa', '')
    text = text.replace('/', '-')
    return text


def load_unigram_set_from_arpa(arpa_path: str) -> Set[str]:
    """Read unigrams from arpa file."""
    unigrams = set()
    with open(arpa_path) as f:
        start_1_gram = False
        for line in f:
            line = line.strip()
            if line == "\\1-grams:":
                start_1_gram = True
            elif line == "\\2-grams:":
                break
            if start_1_gram and len(line) > 0:
                parts = line.split("\t")
                if len(parts) == 3:
                    unigrams.add(parts[1])
    if len(unigrams) == 0:
        raise ValueError("No unigrams found in arpa file. Something is wrong with the file.")
    return unigrams


def evaluate(asr_model_name: str, hypothesis_path: str, lm_model_name: str, output: str):
    processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
    hypothesis = Dataset.load_from_disk(hypothesis_path)

    unigrams = load_unigram_set_from_arpa(lm_model_name.replace('.binary', '.arpa'))
    vocab = {k.lower(): v for k, v in processor.tokenizer.get_vocab().items()}
    del vocab['<s>']
    vocab_list = sorted(vocab.keys(), key=lambda x: vocab[x])
    beam_decoder = build_ctcdecoder(vocab_list, lm_model_name, unigrams=unigrams)

    def map_hypo_to_pred(batch):
        batch['predicted'] = [beam_decoder.decode(i) for i in batch['hypothesis']]
        batch["target"] = batch["sentence"]
        return batch

    result = hypothesis.map(map_hypo_to_pred, batched=True, batch_size=1,
                            remove_columns=list(hypothesis.features.keys()))

    wer = wer_metric.compute(predictions=result["predicted"], references=result["target"])
    prediction_df = pd.DataFrame(result)
    metadata_df = pd.DataFrame.from_dict({
        'asr_model_name': [asr_model_name],
        'hypothesis_path': [hypothesis_path],
        'lm_model_name': [lm_model_name],
        'wer': [wer]
    })

    lm_model_name = clean_model_name(os.path.basename(lm_model_name))
    hypothesis_path = clean_model_name(hypothesis_path)

    filename = f'{hypothesis_path}-{lm_model_name}.xlsx'
    output_file_path = os.path.join(output, filename)
    writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')

    prediction_df.to_excel(writer, sheet_name='Predictions')
    metadata_df.to_excel(writer, sheet_name='Metadata')

    writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Evaluate Language Model')
    parser.add_argument('--asr_model_name', type=str, default='lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2')
    parser.add_argument('--hypothesis_path', type=str, required=True)
    parser.add_argument('--lm_model_name', type=str, required=True,
                        help="Model name on HF Hub or path, or file path for kenlm")
    parser.add_argument('--output', type=str, required=True, help='Output result path')
    args = parser.parse_args()

    evaluate(**vars(args))