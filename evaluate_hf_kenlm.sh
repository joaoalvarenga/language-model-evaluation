#!/bin/bash
DATA_FOLDER=$1
MODELS_FOLDER=$2
OUTPUT_FOLDER=$3
HYPOTHESIS_PATH=$4
GREEN='\e[1;32m'
NC='\033[0m' # No Color

echo ""
#for n in 3 4 5; do
#  for filename in $MODELS_FOLDER/$n-gram/*.arpa; do
#    echo -e "${GREEN}Evaluating $filename...${NC}"
#    python3 evaluate_language_model.py --hypothesis_path $HYPOTHESIS_PATH --lm_model_name $filename --output $OUTPUT_FOLDER
#  done
#done

for filename in $DATA_FOLDER/*.txt; do
  for n in 3 4 5; do
    fbname=$(basename "$filename" | cut -d. -f1)
    model_path="$MODELS_FOLDER/$n-gram/$fbname-$n-gram.arpa"
    echo "Evaluating $model_path"
    python3 evaluate_language_model.py --hypothesis_path $HYPOTHESIS_PATH --lm_model_name $model_path --output $OUTPUT_FOLDER
  done
done