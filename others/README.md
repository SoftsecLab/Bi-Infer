Bi-Infer Baseline Methods

This directory contains implementations of seven baseline methods used for comparison with Bi-Infer in LLM-generated text origin detection experiments.

Baseline Methods

Method

Description

LRR

Measures statistical differences in text using token log-likelihood and log-rank.

Binoculars

Uses two language models to evaluate text through perplexity and cross-perplexity comparisons.

GhostBuster

Extracts statistical features from surrogate language models and performs supervised classification.

SeqXGPT

Uses token-level log-probability sequences with convolutional and self-attention networks for classification.

GPT-Who

Extracts surprisal-based features to distinguish texts from different language models.

Sniffer

Analyzes probability differences and correlations among multiple surrogate models for source identification.

PROFILER

Extracts context-aware inference patterns and uses a classifier to identify text origins.

LRR, Binoculars, and GhostBuster are adapted to multi-class origin detection using a One-vs-Rest (OvR) evaluation strategy.

Installation

Install the required dependencies:

pip install -r requirements.txt

Python 3.10+ and a CUDA-enabled PyTorch environment are recommended.

Dataset Format

Training and test datasets should be provided as JSON files with the following structure:

{
  "human": ["Text 1", "Text 2"],
  "generator_a": ["Text 1", "Text 2"],
  "generator_b": ["Text 1", "Text 2"]
}

All available classes and valid samples are retained without per-class sampling limits.

Usage

Run a single baseline:

python run_exp.py --method ghostbuster --task my_dataset --train data/train.json --test data/test.json --models /path/to/model_A /path/to/model_B --output outputs/ghostbuster --gpu 0

Supported methods:

lrr

binoculars

ghostbuster

seqxgpt

gpt_who

sniffer

profiler

Run all baselines:

python run_exp.py --method all --task my_dataset --train data/train.json --test data/test.json --models /path/to/model_A /path/to/model_B --output outputs --gpu 0

Arguments

Argument

Description

--method

Baseline method name or all.

--task

Custom experiment name.

--train

Path to the training dataset.

--test

Path to the test dataset.

--models

Paths or Hugging Face IDs of surrogate models.

--output

Directory for experimental results.

--gpu

GPU device index.

LRR and Binoculars do not require training data. Binoculars requires two models with compatible tokenizers and vocabularies.

Output

Experimental results, including classification reports and evaluation metrics, are saved in the specified output directory.
