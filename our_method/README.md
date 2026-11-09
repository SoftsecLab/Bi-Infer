# Bi-Infer

Main implementation of Bi-Infer for text origin detection. 

| File | Function |
| --- | --- |
| `config.py` | Model paths and hyperparameters |
| `data.py` | JSON input and adaptive window calculation |
| `feature_extractor.py` | Dual-surrogate feature extraction and concatenation |
| `conformer.py` | Conformer blocks |
| `net.py` | Classifier and masked majority voting |
| `dataset.py` | Feature dataset, padding and masks |
| `train.py` | Training, checkpoint selection and test evaluation |
| `run_pipeline.py` | Complete pipeline |

## Installation

Python 3.10+, CUDA-enabled PyTorch, and locally available surrogate checkpoints:

```bash
pip install -r requirements.txt
```

## Run

From the package root, use this single command:

```bash
python run_pipeline.py --train data/train.json --test data/test.json --qwen /path/to/Qwen2.5-7B-Instruct --llama /path/to/Llama-3.1-8B-Instruct --output outputs --gpu 0
```

Both input files use the same source labels and JSON format:

```json
{"human": ["text ..."], "generator_a": ["text ..."]}
```

The command extracts features, trains the classifier, and evaluates the best checkpoint. Outputs are saved under `outputs/features/` and `outputs/results/`.
