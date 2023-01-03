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

for n in 3 4 5; do
    for beam_width in 10 50; do
      fbname=$(basename "$filename" | cut -d. -f1)
      model_path="$MODELS_FOLDER/$n-gram/$fbname-$n-gram.binary"
      echo "Evaluating $n-gram $beam_width"
      python3 evaluate_language_model_multiple.py --beam_width $beam_width --hypothesis_path $HYPOTHESIS_PATH --models_folder $MODELS_FOLDER --order $n --data_folder $DATA_FOLDER --output $OUTPUT_FOLDER
    done
done