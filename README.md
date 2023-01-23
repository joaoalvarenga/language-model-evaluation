# Language Model Evaluation for Automatic Speech Recognition

Resources and extra documentation for the manuscript "Data-Centric Approach for Portuguese Speech Recognition: Language Model And Its Implications" published in IEEE Latin America Transactions.

![Graphical abstract](/docs/graphical-abstract.png)


## Data

### Wikipedia Dump
A wikipedia dump from 2018
Download - http://www02.smt.ufrj.br/~igor.quintanilha/ptwiki-20181125.txt

### CETUC
Contains approximately 145 hours of Brazilian Portuguese speech
distributed among 50 male and 50 female speakers, each pronouncing approximately 1,000 phonetically balanced sentences selected from the CETENFolha6
corpus;

Download - http://www02.smt.ufrj.br/~igor.quintanilha/alcaim.tar.gz

### Common Voice
A project proposed by Mozilla Foundation with the goal to create a wide open dataset in different languages. In this project, volunteers donate and validate speech using the official site

Version 8.0

Download - https://commonvoice.mozilla.org/pt/datasets

### CORAA
CORAA is a publicly available dataset for Automatic Speech Recognition (ASR) in the Brazilian Portuguese language containing 290.77 hours of audios and their respective transcriptions (400k+ segmented audios). 

Version 1.1
Download - https://github.com/nilc-nlp/CORAA

### MLS

Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish.

Download - http://www.openslr.org/94/


### Install depenencies
First you need to manually install KenLM compiling from the instructions [here](https://github.com/kpu/kenlm).
Then you can just run
```
poetry install
```

### Usage
You can use `generate_hypothesis.py` to generate Wav2Vec2 hypothesis for decoding.
```
python3 generate_hypothesis.py \
    --data_type commonvoice \
    --data_folder ./data/cv-corpus-6.1-2020-12-11/pt/ \
    --model_name ./wav2vec2-pt-cv-6.1-coraa \
    --output_path ./hypothesis/cv-6.1-w2v-cv-6.1-coraa \
    --device cuda
```
Now you can use `combine_datasets.py` to generate combinations of all datasets and estimate KenLM variations using `estimate_kenlm.sh`.

Then you can use `evaluate_hf_kenlm_multiple.sh` to decode hypothesis varying some n-grams parameters and generate CSV with outputs.
