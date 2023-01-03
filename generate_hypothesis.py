import argparse
import os

parser = argparse.ArgumentParser(prog='Generate hypothesis')
parser.add_argument('--data_type', type=str, required=True, help='Dataset Type')
parser.add_argument('--data_folder', type=str, required=True, help='Common Voice folder, ex: ./data/cv-corpus-6.1-2020-12-11/pt/')
parser.add_argument('--model_name', type=str)
parser.add_argument('--output_path', type=str, required=True, help='Output hypothesis path')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

import torchaudio

def load_data_cv(batch):
    clips_path = os.path.join(args.data_folder, 'clips')
    full_path = os.path.join(clips_path, batch['path'])
    speech, _ = torchaudio.load(full_path)
    return speech.squeeze(0).numpy()


def load_data_coraa(batch):
    full_path = os.path.join(args.data_folder, batch['path'])
    speech, _ = torchaudio.load(full_path)
    return speech.squeeze(0).numpy()

datasets_type = {
    'commonvoice': load_data_cv,
    'coraa': load_data_coraa
}



from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import os

import torch
import re
import sys

chars_to_ignore_regex = '[\,\?\.\!\;\:\"]'  # noqa: W605
wer = load_metric("wer")
model = Wav2Vec2ForCTC.from_pretrained(args.model_name).to(args.device)
processor = Wav2Vec2Processor.from_pretrained(args.model_name)

def map_to_pred(batch):
    features = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0], padding=True, return_tensors="pt")
    input_values = features.input_values.to(args.device)
    attention_mask = features.attention_mask.to(args.device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["hypothesis"] = logits.cpu().detach().numpy()
    batch["predicted"] = processor.batch_decode(pred_ids)
    batch["predicted"] = [pred.lower() for pred in batch["predicted"]]
    batch['target'] = batch['sentence']
    return batch

dataset = load_dataset('csv', split='test', data_files={'test': args.data_folder + 'test.csv'}, keep_in_memory=True)

load_data_fn = datasets_type[args.data_type.lower()]

def map_to_array(batch):
    batch["speech"] = load_data_fn(batch) #resampler.forward(speech.squeeze(0)).numpy()
    batch["sampling_rate"] = resampler.new_freq
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower().replace("’", "'")
    return batch

ds = dataset.map(map_to_array, keep_in_memory=True)
result = ds.map(map_to_pred, batched=True, batch_size=1, remove_columns=list(ds.features.keys()), keep_in_memory=True)

result.save_to_disk(args.output_path)
