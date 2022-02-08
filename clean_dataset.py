import argparse
from tqdm import tqdm
from asr_language_model_evaluation.datasets import BRWaC, Wikipedia, Coraa, CommonVoice, CETUC, MultilingualLibriSpeech, \
    CetenFolha
from asr_language_model_evaluation.preprocessing import normalize


datasets_type = {
    'brwac': BRWaC,
    'wikipedia': Wikipedia,
    'coraa': Coraa,
    'commonvoice': CommonVoice,
    'cetuc': CETUC,
    'mls': MultilingualLibriSpeech,
    'cetenfolha': CetenFolha
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Clean dataset')
    parser.add_argument("--path", type=str, required=True,
                        help="Dataset file path")
    parser.add_argument('--type', type=str, required=True, help='Dataset type (BRWaC, Wikipedia, CORAA, CommonVoice, CETUC)')
    parser.add_argument('--output', type=str, required=True, help='Dataset output file path')
    args = parser.parse_args()

    dataset_type = args.type.lower()
    assert dataset_type in datasets_type.keys()

    dataset = datasets_type[dataset_type](args.path)
    with open(args.output, 'w') as f:
        for data in tqdm(dataset):
            normalized_text = normalize(data['text'])
            f.write(f"{normalized_text}\n")
            #f.write(f"{data['text']}\n")

