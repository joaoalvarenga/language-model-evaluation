#!/bin/bash
DATA_FOLDER=$1
MODELS_FOLDER=$2
OUTPUT_FOLDER=$3
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\e[1;32m'
RED='\e[0;31m'

echo ""
mkdir -p "$OUTPUT_FOLDER/3-gram"
mkdir -p "$OUTPUT_FOLDER/4-gram"
mkdir -p "$OUTPUT_FOLDER/5-gram"
for filename in $DATA_FOLDER/*.txt; do
  for n in 3 4 5; do
    echo "Training $n-gram for $filename"
    fbname=$(basename "$filename" | cut -d. -f1)
    model_path="$MODELS_FOLDER/$n-gram/$fbname-$n-gram.arpa"
    output_path="$OUTPUT_FOLDER/$n-gram/$fbname-$n-gram.lst"
    echo "Estimating to $output_path"
    if [ -f $output_path ]; then
       echo -e "${YELLOW}Model $output_path already exists in $real_output_path ${NC}"
       continue
    fi
    #echo -e "${RED}Mocking estimating for $output_path ${NC}"
    python3 extract_unigrams_from_arpa.py --model_path $model_path --output $output_path
    # $KENLM_ESTIMATION -o $n < $filename > $output_path
  done
done
