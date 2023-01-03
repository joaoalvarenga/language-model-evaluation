#!/bin/bash
KENLM_FOLDER=$1 #/media/work/joaoalvarenga/language_models/kenlm/build/bin
DATA_FOLDER=$2
MODELS_FOLDER=$3
OUTPUT_FOLDER=$4
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\e[1;32m'
RED='\e[0;31m'

KENLM_BINARY=$KENLM_FOLDER/build_binary
echo ""
mkdir -p "$OUTPUT_FOLDER/3-gram"
mkdir -p "$OUTPUT_FOLDER/4-gram"
mkdir -p "$OUTPUT_FOLDER/5-gram"
for filename in $DATA_FOLDER/*.txt; do
  for n in 3 4 5; do
    echo "Training $n-gram for $filename"
    fbname=$(basename "$filename" | cut -d. -f1)
    model_path="$MODELS_FOLDER/$n-gram/$fbname-$n-gram.arpa"
    output_path="$OUTPUT_FOLDER/$n-gram/$fbname-$n-gram.binary"
    echo "Estimating to $output_path"
    if [ -f $output_path ]; then
       echo -e "${YELLOW}Model $output_path already exists${NC}"
       continue
    fi
    #echo -e "${RED}Mocking estimating for $output_path ${NC}"
    $KENLM_BINARY $model_path $output_path
  done
done
