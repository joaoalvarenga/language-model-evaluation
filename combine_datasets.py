import argparse
import itertools as it
import glob
import os

from tqdm import tqdm


def load_file(filename):
    with open(filename) as f:
        data = set([l.strip() for l in f])
    return data


def clean_dataset_name(name: str) -> str:
    name = name.replace('.txt', '')
    name = name.replace('.', '-')
    return name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Clean dataset')
    parser.add_argument("--path", type=str, required=True,
                        help="Normalized dataset folder")
    parser.add_argument('--output', type=str, required=True, help='Combinations output folder')
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.path, '*'))
    for i in range(len(files)):
        combinations = list(it.combinations(files, i + 1))
        print('Combining', i + 1, 'Total combinations:', len(combinations))
        for combination in tqdm(combinations, total=len(combinations)):
            filenames = sorted([os.path.basename(c) for c in combination])
            data = set()
            final_name = '-'.join([clean_dataset_name(name) for name in filenames])
            if os.path.exists(os.path.join(args.output, f'{final_name}.txt')):
                print('Skipping', final_name)
                continue
            for filename in filenames:
                data = data.union(load_file(os.path.join(args.path, filename)))
            with open(os.path.join(args.output, f'{final_name}.txt'), 'w') as f:
                f.write('\n'.join(data))
