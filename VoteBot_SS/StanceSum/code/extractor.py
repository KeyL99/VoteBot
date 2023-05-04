import numpy as np
import math
import torch
import random

from utils import evaluate_f1
from bert_model import BertEdgeScorer, BertConfig


class PacSumExtractor:

    def __init__(self, extract_num=3, beta=3, lambda1=-0.2, lambda2=-0.2):

        self.extract_num = extract_num
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def extract_summary(self, data_iterator):

        summaries = []
        model_tags = []
        references = []
        sen_num_per_doc = []


        for idx, item in enumerate(data_iterator):
            data_dict = {}
            article, abstract, issue_sim, intent_prob, inputs = item

            sen_num_per_doc.append(len(article))

            if len(article) <= self.extract_num:
                summaries.append(article)
                model_tags.append([x for x in range(len(article))])
                references.append(abstract)

                data_dict['article'] = article
                data_dict['target'] = abstract
                data_dict['predict'] = [x for x in range(len(article))]
                data_dict['length'] = len(article)

                continue

            edge_scores = self._calculate_similarity_matrix(*inputs)
            ids = self._select_tops(edge_scores, issue_sim, intent_prob, beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)
            summary = list(map(lambda x: article[x], ids))
            summaries.append(summary)
            model_tags.append(ids)
            references.append(abstract)

            data_dict['article'] = article
            data_dict['target'] = abstract
            data_dict['predict'] = ids
            data_dict['length'] = len(article)

        result = evaluate_f1(model_tags, references, sen_num_per_doc)
        print(result)


    def tune_hparams(self, data_iterator):

        pred, true = [], []
        sen_num_per_doc = []
        for idx, item in enumerate(data_iterator):
            article, abstract, issue_sim, intent_prob, inputs = item
            # article, abstract, issue_sim, inputs = item

            if len(article) < 2:
                continue
            edge_scores = self._calculate_similarity_matrix(*inputs)
            tops_list, hparam_list = self._tune_extractor(edge_scores, issue_sim, intent_prob)

            sen_num_per_doc.append(len(article))
            pred.append(tops_list)
            true.append(abstract)
        best_f1 = 0
        best_hparam = None
        for i in range(len(pred[0])):
            result = evaluate_f1([pred[k][i] for k in range(len(pred))], true, sen_num_per_doc)

            if result['f1'] > best_f1:
                best_f1 = result['f1']
                best_hparam = hparam_list[i]

        print("The best hyper-parameter :  beta %.4f , lambda1 %.4f, lambda2 %.4f " % (best_hparam[0], best_hparam[1], best_hparam[2]))
        print("The best rouge_1_f_score :  %.4f " % best_f1)

        self.beta = best_hparam[0]
        self.lambda1 = best_hparam[1]
        self.lambda2 = best_hparam[2]

    def _calculate_similarity_matrix(self, *inputs):

        raise NotImplementedError

    def _select_tops(self, edge_scores, issue_sim, intent_prob, beta, lambda1, lambda2):

        min_score = edge_scores.min()
        max_score = edge_scores.max()

        edge_threshold = min_score + beta * (max_score - min_score)
        new_edge_scores = edge_scores - edge_threshold
        forward_scores, backward_scores, _ = self._compute_scores(new_edge_scores, 0)

        forward_scores = 0 - forward_scores

        scores = [float(0) for _ in range(len(new_edge_scores))]
        for idx in range(len(scores)):
            scores[idx] = float(forward_scores[idx] + backward_scores[idx])

        scores = torch.sigmoid(torch.from_numpy(np.array(scores))).numpy().tolist()

        new_issue_sim = issue_sim
        new_intent_prob = intent_prob

        paired_scores = []
        for node in range(len(forward_scores)):
            is_sim = new_issue_sim[node]
            in_prob = new_intent_prob[node]
            ss = (lambda1 * scores[node] + lambda2 * is_sim)

            if ss <= 0:
                paired_scores.append([node, in_prob])
            else:
                paired_scores.append([node, ss * in_prob])

        #shuffle to avoid any possible bias
        random.shuffle(paired_scores)
        paired_scores.sort(key=lambda x: x[1], reverse=True)
        extracted = [item[0] for item in paired_scores[:self.extract_num]]

        return extracted

    def _compute_scores(self, similarity_matrix, edge_threshold):

        forward_scores = [0 for i in range(len(similarity_matrix))]
        backward_scores = [0 for i in range(len(similarity_matrix))]

        edges = []
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix[i])):
                edge_score = similarity_matrix[i][j]
                if edge_score > edge_threshold:
                    forward_scores[j] += edge_score
                    backward_scores[i] += edge_score
                    edges.append((i, j, edge_score))

        return np.asarray(forward_scores), np.asarray(backward_scores), edges


    def _tune_extractor(self, edge_scores, issue_sim, intent_prob):

        tops_list = []
        hparam_list = []
        num = 10
        for k in range(num + 1):
            beta = k / num
            for i in range(11):
                lambda1 = i/10
                lambda2 = 1 - lambda1
                extracted = self._select_tops(edge_scores, issue_sim, intent_prob, beta=beta, lambda1=lambda1, lambda2=lambda2)

                tops_list.append(extracted)
                hparam_list.append((beta, lambda1, lambda2))

        return tops_list, hparam_list


class PacSumExtractorWithBert(PacSumExtractor):

    def __init__(self, bert_model_file, bert_config_file, extract_num=3, beta=3, lambda1=-0.2, lambda2=-0.2):

        super(PacSumExtractorWithBert, self).__init__(extract_num, beta, lambda1, lambda2)
        self.model = self._load_edge_model(bert_model_file, bert_config_file)

    def _calculate_similarity_matrix(self,  x, t, w, x_c, t_c, w_c, pair_indice):
        #doc: a list of sequences, each sequence is a list of words

        def pairdown(scores, pair_indice, length):
            #1 for self score
            out_matrix = np.ones((length, length))
            for pair in pair_indice:
                out_matrix[pair[0][0]][pair[0][1]] = scores[pair[1]]
                out_matrix[pair[0][1]][pair[0][0]] = scores[pair[1]]

            return out_matrix

        scores = self._generate_score(x, t, w, x_c, t_c, w_c)
        doc_len = int(math.sqrt(len(x)*2)) + 1
        similarity_matrix = pairdown(scores, pair_indice, doc_len)

        return similarity_matrix

    def _generate_score(self, x, t, w, x_c, t_c, w_c):

        #score =  log PMI -log k
        scores = torch.zeros(len(x)).cuda()
        step = 20
        for i in range(0, len(x), step):

            batch_x = x[i:i+step]
            batch_t = t[i:i+step]
            batch_w = w[i:i+step]
            batch_x_c = x_c[i:i+step]
            batch_t_c = t_c[i:i+step]
            batch_w_c = w_c[i:i+step]

            inputs = tuple(t.to('cuda') for t in (batch_x, batch_t, batch_w, batch_x_c, batch_t_c, batch_w_c))
            batch_scores, batch_pros = self.model(*inputs)
            scores[i:i+step] = batch_scores.detach()


        return scores

    def _load_edge_model(self, bert_model_file, bert_config_file):

        bert_config = BertConfig.from_json_file(bert_config_file)
        model = BertEdgeScorer(bert_config)
        model_states = torch.load(bert_model_file)
        model.bert.load_state_dict(model_states)

        model.cuda()
        model.eval()
        return model
