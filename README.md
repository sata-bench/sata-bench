# SATA-Bench
<img src="imgs/sata_llamas.png" width="60%">
This repo contains code for SATA-Bench including dataset, evaluation scripts, and methods implementation.

## 🗂️ Repo Structure
```
src/satabench/
├── evaluation/           # scripts for evaluating LLMs in paper Section3
│   ├── dataset/          # contains a test dataset to use
│   └── metrics/          # metrics implementation
|
├── methods/              # methods implementation in paper Section4
│   ├── choice_funnel/    # Choice funnel algorithm
│   ├── data/             # contains a test dataset to use
│   └── utils/            # utility functions
```
## 🚀 Quick Start

Tested Python version: python==3.10.
```bash
# Clone the repository
git clone https://github.com/sata-bench/sata-bench.git
cd sata-bench

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Evaluate SATA-Bench using probability based retrieval methods:
Run below command to launch SATA-Bench evaluation job.   
By default it will use ```Qwen/Qwen2.5-14B-Instruct``` model using "Choice Funnel" method with 200 samples (see below for job configuration)
```
bash src/satabench/methods/run.sh
```
Once finished you'll see output like:
```
Computing final metrics...

        Metrics Summary:
        EM (Exact Match): 0.275000 ↑
        Precision: 77.04% ↑
        Recall: 58.51% ↑
        JI (Jaccard Index): 54.86% ↑
        SPD: 3.84 ↓
        RStd: 8.74 ↓
        RSD: 0.76 ↓
        CtDif (Count Difference): -1.35
        CtDifAbs (Absolute Count Difference): 1.56 ↓
        CtAcc (Count Accuracy): 0.32 ↑
        InfCost (Inference Cost): 1857 ↓
        

Evaluation completed!
Total execution time: 0h 3m 53.71s
Done! - Execution Success for Model: Qwen/Qwen2.5-14B-Instruct
```
A local file named ```$MODEL_NAME_result.txt``` will be stored with resutls for reference.

🔧  More Detailed Job Configuration:

Set env variable ```export MODEL_NAME``` to select target model from list below:
```
# Models evaluated in the paper:
model_names = ["mistralai/Ministral-8B-Instruct-2410",      #0
                "bigscience/bloomz-7b1",                    #1
                "microsoft/Phi-3-small-8k-instruct",        #2
                "meta-llama/Meta-Llama-3-8B-instruct",      #3
                "google/gemma-7b-it",                       #4
                "Qwen/Qwen2.5-14B-Instruct",                #5
                "microsoft/Phi-4-mini-reasoning",           #6
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"] #7
```
Set env variable ```export METHOD``` to select method from below:
```
1) "choice_funnel":         Proposed method "Choice Funnel" in SATA-Bench Paper Section4
2) "first_token":           Baseline Method #1 in paper table4
3) "first_token_debiasing": Baseline Method #2 in paper table4
4) "yesno":                 Baseline Method #3 in paper table4
```
Set env variable ```export SAMPLE_COUNT``` to adjust the sample count of SATA-Bench based on needed (max 1650).


## 🔍 Issue Reporting

If you encounter any bugs, have feature requests, or experience other issues with SataBench, please report them by opening an issue in our [GitHub Issues]([https://github.com/yourusername/satabench/issues](https://github.com/sata-bench/sata-bench/issues)) page. 

When reporting an issue, please include:
- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Any relevant logs or error messages
- Your environment information (OS, Python version, etc.)

This helps us address your concerns more efficiently and improve SataBench for everyone.
