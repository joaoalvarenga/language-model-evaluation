#!/bin/bash
KENLM_FOLDER=$1
DATA_FOLDER=$2
OUTPUT_FOLDER=$3

KENLM_ESTIMATION=$KENLM_FOLDER/lmplz
echo ""
mkdir -p "$OUTPUT_FOLDER/3-gram"
mkdir -p "$OUTPUT_FOLDER/4-gram"
mkdir -p "$OUTPUT_FOLDER/5-gram"
for filename in $DATA_FOLDER/*.txt; do
  for n in 3 4 5; do
    echo "Training $n-gram for $filename"
    fbname=$(basename "$filename" | cut -d. -f1)
    $KENLM_ESTIMATION -o $n < $filename > "$OUTPUT_FOLDER/$n-gram/$fbname-$n-gram.arpa"
  done
done