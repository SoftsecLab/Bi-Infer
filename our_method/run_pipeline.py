"""Main pipeline: feature extraction, classifier training and test evaluation."""
import argparse
from pathlib import Path
from .config import MODEL_PATHS
from .feature_extractor import extract_features
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, help='Training JSON: label -> list of texts')
    parser.add_argument('--test', required=True, help='Test JSON: label -> list of texts')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--qwen', default=MODEL_PATHS['qwen7'])
    parser.add_argument('--llama', default=MODEL_PATHS['llama8'])
    args = parser.parse_args()
    output = Path(args.output)
    model_paths = {'qwen7': args.qwen, 'llama8': args.llama}
    train_cache, test_cache = extract_features(
        args.train, args.test, output / 'features', model_paths, f'cuda:{args.gpu}')
    train(train_cache, test_cache, output / 'results', args.gpu)


if __name__ == '__main__':
    main()
