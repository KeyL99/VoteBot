# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pickle as pkl

PAD, CLS = '[PAD]', '[CLS]'
MAX_VOCAB_SIZE = 10000
UNK = '<UNK>'

def build_vocab(file_path2, file_path3, tokenizer, max_size, min_freq):
    vocab_dic2 = {}  # id.txt
    vocab_dic3 = {}  # id2.txt
    with open(file_path2, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic2[word] = vocab_dic2.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic2.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic2 = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic2.update({UNK: len(vocab_dic2), PAD: len(vocab_dic2) + 1})

    with open(file_path3, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic3[word] = vocab_dic3.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic3.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic3 = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic3.update({UNK: len(vocab_dic3), PAD: len(vocab_dic3) + 1})

    vocab_dic2.update(vocab_dic3)

    return vocab_dic2


def build_dataset(config):
    tokenizer = lambda x: x.split(' ')
    vocab_id = build_vocab(config.id_path, config.id2_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)

    pkl.dump(vocab_id, open(config.vocab_id_path, 'wb'))
    print(f"vocab_id size: {len(vocab_id)}")

    def load_dataset(path, pad_size=128):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    def load_id(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content = lin
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab_id.get(word, vocab_id.get(UNK)))
                contents.append((words_line, seq_len, 'cls'))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    id = load_id(config.id_path, config.pad_size)
    id2 = load_id(config.id2_path, config.pad_size)

    return vocab_id, train, dev, test, id, id2


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
