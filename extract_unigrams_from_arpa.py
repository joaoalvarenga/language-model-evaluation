import argparse
from typing import Set


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Extract Unigrams from Arpa')
    parser.add_argument('--model_path', type=str, required=True, help='LM arpa path')
    parser.add_argument('--output', type=str, required=True, help='Output unigrams file')
    args = parser.parse_args()
    
    unigrams = load_unigram_set_from_arpa(args.model_path)
    
    with open(args.output, 'w') as f:
        f.write('\n'.join(unigrams))