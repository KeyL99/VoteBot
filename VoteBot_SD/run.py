# coding: UTF-8
import json
import os
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

type_list = ['-1', '0', '1']
type2id = {type: i for i, type in enumerate(type_list)}

def read_json(input_file):
    lines = []
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            text = ' '.join(line['tokens'])
            label = line['label']
            words = text.split()

            lines.append({"words": words, "label": label})
    return lines

def read_json_ID(input_file):
    lines = []
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            text3 = line['character'] + ' ' + line['reply'] + ' ' + line['beReplied']
            lines.append(text3)
    return lines

def preprocess(data):
    corpus, labels = [], []

    for sent_map in data:
        corpus.append(' '.join(sent_map['words']))
        labels.append(type2id[sent_map['label']])
    return corpus, labels

def write_class(filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for type in type_list:
            f.write(type + '\n')

def write_data(filename, x, y):
    with open(filename, 'w', encoding='utf-8') as f:
        for text, label in zip(x, y):
            line = text + '\t' + str(label) + '\n'
            f.write(line)

def write_id(filename, x):
    with open(filename, 'w', encoding='utf-8') as f:
        # for text in zip(x):
        for text in x:
            line = str(text) + '\n'
            f.write(line)

if __name__ == '__main__':
    dataset = 'data/classification'
    DATA_PATH = 'data/classification'

    total_p1 = 0
    total_r1 = 0
    total_f1 = 0

    PATH = './data'
    train_data = read_json(os.path.join(PATH, 'train.json'))
    test_data = read_json(os.path.join(PATH, 'test.json'))

    ID = read_json_ID(os.path.join(PATH, 'train.json'))
    ID2 = read_json_ID(os.path.join(PATH, 'test.json'))

    train_x, train_y = preprocess(train_data)
    test_x, test_y = preprocess(test_data)

    write_data(os.path.join(DATA_PATH, 'data', 'train.txt'), train_x, train_y)
    write_data(os.path.join(DATA_PATH, 'data', 'test.txt'), test_x, test_y)
    write_data(os.path.join(DATA_PATH, 'data', 'dev.txt'), test_x, test_y)
    write_class(os.path.join(DATA_PATH, 'data', 'class.txt'))
    write_id(os.path.join(DATA_PATH, 'data', 'id.txt'), ID)
    write_id(os.path.join(DATA_PATH, 'data', 'id2.txt'), ID2)

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    vocab_id, train_data, dev_data, test_data, id, id2 = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    config.n_vocab_id = len(vocab_id)

    # train
    model = x.Model(config).to(config.device)
    fold_num = 0
    p1, r1 = train(config, model, train_iter, dev_iter, test_iter, id, id2, fold_num)
    f1 = 2 * p1 * r1 / (p1 + r1)

    total_p1 += p1
    total_r1 += r1
    total_f1 += f1

    print('p: ', p1)
    print('r: ', r1)
    print('f1: ', f1)


