#!/bin/bash
MODELS_FOLDER=$1
OUTPUT_FOLDER=$2
HYPOTHESIS_PATH=$3
GREEN='\e[1;32m'
NC='\033[0m' # No Color

echo ""
for n in 3 4 5; do
  for filename in $MODELS_FOLDER/$n-gram/*.arpa; do
    echo -e "${GREEN}Evaluating $filename...${NC}"
    python3 evaluate_language_model.py --hypothesis_path $HYPOTHESIS_PATH --lm_model_name $filename --output $OUTPUT_FOLDER
  done
done