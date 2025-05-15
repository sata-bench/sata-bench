#!/bin/bash
# Models evaluated in the paper:
# model_names = ["mistralai/Ministral-8B-Instruct-2410", #0
# "bigscience/bloomz-7b1", #1
# "microsoft/Phi-3-small-8k-instruct", #2
# "meta-llama/Meta-Llama-3-8B-instruct", #3
# "google/gemma-7b-it", #4
# "Qwen/Qwen2.5-14B-Instruct", #5
# "microsoft/Phi-4-mini-reasoning", #6
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"] #7
#
# Attention:
# Some of the models require users to sign up for approval on
# Huggingface (e.g., llama, mistral). To test those model locally
# please get approval first and set your HF_TOKEN as an env
# var in this file: HF_TOKEN="{your_hf_token}"

# Set default MODEL_NAME if not already set by user
if [ -z "$MODEL_NAME" ]; then
  export MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
  echo "Using default model: $MODEL_NAME"
else
  echo "Using user-specified model: $MODEL_NAME"
fi

# Set default SAMPLE_COUNT if not already set by user
if [ -z "$SAMPLE_COUNT" ]; then
  export SAMPLE_COUNT=200
  echo "Using default sample count: $SAMPLE_COUNT"
else
  echo "Using user-specified sample count: $SAMPLE_COUNT"
fi

# Set default METHOD if not already set by user
if [ -z "$METHOD" ]; then
  export METHOD="choice_funnel"
  echo "Using default method: $METHOD"
else
  echo "Using user-specified method: $METHOD"
fi

# Available Methods:
# 1) "choice_funnel": Proposed method "Choice Funnel" in SATA-Bench Paper Section4
# 2) "first_token": Baseline Method #1 in paper table4
# 3) "first_token_debiasing": Baseline Method #2 in paper table4
# 4) "yesno": Baseline Method #3 in paper table4

echo "Running for Model: $MODEL_NAME"
STORED_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/.*\///')
OUTPUT_FILE="$(pwd)/${STORED_MODEL_NAME}_result.txt"
echo "Output will be saved to: $OUTPUT_FILE"

# Run the script with output to both terminal and file
python3 -m src.satabench.methods.choice_funnel.sata_scoring_executor | tee "${STORED_MODEL_NAME}_result.txt"

# Check if the execution was successful
if [ $? -eq 0 ]; then
  echo "Done! - Execution Success for Model: $MODEL_NAME"
else
  echo "Error! - Execution Failed for Model: $MODEL_NAME"
  exit 1
fi