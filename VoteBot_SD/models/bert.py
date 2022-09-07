# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.vocab_id_path = dataset + '/data/vocab_id.pkl'                          # id词表
        self.id_path = dataset + '/data/id.txt'
        self.id2_path = dataset + '/data/id2.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                              # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                                      # 类别数
        self.num_epochs = 50                                                         # epoch数
        self.batch_size = 32                                                         # mini-batch大小
        self.pad_size = 128                                                          # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                                    # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.embeddingID = nn.Embedding(config.n_vocab_id, 130, padding_idx=config.n_vocab_id - 1)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size + 390, config.num_classes)

    def forward(self, x):

        characterList = []
        replyList = []
        beRepliedList = []

        for i in range(len(x[3])):
            characterList.append(x[3][i][0])
            replyList.append(x[3][i][1])
            beRepliedList.append(x[3][i][2])

        character_numpy = np.array(characterList)
        character_tensor = torch.from_numpy(character_numpy).cuda()

        reply_numpy = np.array(replyList)
        reply_tensor = torch.from_numpy(reply_numpy).cuda()

        beReplied_numpy = np.array(beRepliedList)
        beReplied_tensor = torch.from_numpy(beReplied_numpy).cuda()

        character = self.embeddingID(character_tensor)  # 32*x0
        reply = self.embeddingID(reply_tensor)  # 32*x0
        beReplied = self.embeddingID(beReplied_tensor)  # 32*x0




        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        y = torch.cat([pooled, character, reply, beReplied], dim=1)

        out = self.fc(y)
        return out
