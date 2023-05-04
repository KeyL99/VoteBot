from extractor import PacSumExtractorWithBert
from data_iterator import Dataset

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='tune', choices=['tune', 'test'], help='tune or test')
    parser.add_argument('--rep', type=str, default='bert', choices=['bert'], help='bert')
    parser.add_argument('--extract_num', type=int, default=3, help='number of extracted sentences')
    parser.add_argument('--bert_config_file', type=str, default='../../models/config.json',
                        help='bert configuration file')
    parser.add_argument('--bert_model_file', type=str, default='../../models/pytorch_model.bin', help='bert model file')
    parser.add_argument('--bert_vocab_file', type=str, default='../../models/vocab.txt', help='bert vocabulary file')

    parser.add_argument('--beta', type=float, default=0, help='beta')
    parser.add_argument('--lambda1', type=float, default=0.4, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.6, help='lambda2')

    parser.add_argument('--tune_data_file', type=str, default='../../data/train/*.json',
                        help='data for tunining hyperparameters')
    parser.add_argument('--test_data_file', type=str, default='../../data/test/*.json',
                        help='data for testing')

    args = parser.parse_args()

    extractor = PacSumExtractorWithBert(bert_model_file=args.bert_model_file,
                                        bert_config_file=args.bert_config_file,
                                        extract_num=1,
                                        beta=args.beta,
                                        lambda1=args.lambda1,
                                        lambda2=args.lambda2)
    # tune
    if args.mode == 'tune':
        tune_dataset = Dataset(args.tune_data_file, vocab_file=args.bert_vocab_file)
        tune_dataset_iterator = tune_dataset.iterate_once_doc_bert()
        extractor.tune_hparams(tune_dataset_iterator)

    # test
    test_dataset = Dataset(args.test_data_file, vocab_file=args.bert_vocab_file)
    test_dataset_iterator = test_dataset.iterate_once_doc_bert()
    extractor.extract_summary(test_dataset_iterator)


if __name__ == '__main__':
        main()
