import argparse
import os


def clean_model_name(text: str) -> str:
    text = text.replace('.binary', '')
    text = text.replace('.arpa', '')
    text = text.replace('/', '-')
    return text


def get_output_path(lm_model_name: str, hypothesis_path: str, output: str, beam_width: int):
    lm_model_name = clean_model_name(os.path.basename(lm_model_name))
    hypothesis_path = clean_model_name(hypothesis_path)

    filename = f'{hypothesis_path}-{lm_model_name}-{beam_width}.xlsx'
    return os.path.join(output, filename)


def already_exists(lm_model_name, hypothesis_path, output, beam_width):
    output_file_path = get_output_path(lm_model_name, hypothesis_path, output, beam_width)
    if os.path.exists(output_file_path):
        print('Already exists ', output_file_path, '. Skipping...')
        quit()


parser = argparse.ArgumentParser(prog='Evaluate Language Model')
parser.add_argument('--asr_model_name', type=str)
parser.add_argument('--hypothesis_path', type=str, required=True)
parser.add_argument('--lm_model_name', type=str, required=True,
                    help="Model name on HF Hub or path, or file path for kenlm")
parser.add_argument('--output', type=str, required=True, help='Output result path')
parser.add_argument('--beam_width', type=int, default=100)
args = parser.parse_args()

if args.lm_model_name.find('cv-corpus-7-0') > -1:
    print('Skipping cv-7-0')
    quit()
already_exists(args.lm_model_name, args.hypothesis_path, args.output, args.beam_width)

import numpy as np
import pandas as pd

from typing import Set
from datasets import load_dataset, load_metric, Dataset
from transformers import Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

chars_to_ignore_regex = '[\,\?\.\!\;\:\"]'  # noqa: W605
wer_metric = load_metric("wer")


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


def evaluate(asr_model_name: str, hypothesis_path: str, lm_model_name: str, output: str, beam_width: int):
    output_file_path = get_output_path(lm_model_name, hypothesis_path, output, beam_width)
    if os.path.exists(output_file_path):
        print('Already exists ', output_file_path, '. Skipping...')
        return
    print('Writing to', output_file_path)
    processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
    hypothesis = Dataset.load_from_disk(hypothesis_path)

    unigrams = load_unigram_set_from_arpa(lm_model_name.replace('.binary', '.arpa'))
    vocab = {k.lower(): v for k, v in processor.tokenizer.get_vocab().items()}
    del vocab['<s>']
    vocab_list = sorted(vocab.keys(), key=lambda x: vocab[x])
    beam_decoder = build_ctcdecoder(vocab_list, lm_model_name, unigrams=unigrams)

    def map_hypo_to_pred(batch):
        batch['predicted'] = [beam_decoder.decode(np.asarray(i), beam_width=beam_width) for i in batch['hypothesis']]
        return batch

    result = hypothesis.map(map_hypo_to_pred, batched=True, batch_size=1,
                            remove_columns=['hypothesis'], keep_in_memory=True)

    wer = wer_metric.compute(predictions=result["predicted"], references=result["target"])
    prediction_df = pd.DataFrame(result)
    metadata_df = pd.DataFrame.from_dict({
        'metadata': ['asr_model_name', 'hypothesis_path', 'lm_model_path', 'wer'],
        'value': [asr_model_name, hypothesis_path, lm_model_name, wer]
    })

    writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')

    prediction_df.to_excel(writer, sheet_name='Predictions')
    metadata_df.to_excel(writer, sheet_name='Metadata')

    writer.save()


if __name__ == '__main__':
    evaluate(**vars(args))
