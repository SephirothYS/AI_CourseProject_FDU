
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 500
HIDDEN_DIM = 400
EPOCH = 1

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix["未知"])
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
        
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
                torch.randn(2, 1, self.hidden_dim // 2).to(self.device))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas
        
        forward_var = forward_var.to(self.device)

        for feat in feats:
            alphas_t = []  
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        
        tags = tags.to(self.device)
        t = torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long)
        t = t.to(self.device)
        tags = torch.cat([t, tags])
        score = score.to(self.device)
        
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):   
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        forward_var = forward_var.to(self.device)
        for feat in feats:
            bptrs_t = []  
            viterbivars_t = []  

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    
def get_data(filepath):
    fp = open(filepath,"r",encoding="utf-8")
    words = []
    tags = []
    train_data = []
    for line in fp:
        if line == "\n":
            senten = copy.deepcopy((words,tags))
            train_data.append(senten)
            words.clear()
            tags.clear()
            continue
        items = line.split()
        word, tag = items[0], items[1].rstrip()
        words.append(word)
        tags.append(tag)
    # if words.__len__ != 0:
        # senten = copy.deepcopy((words,tags))
        # train_data.append(senten)
    return train_data

def get_word_ix(train_data):

    word_to_ix = {}
    for sentence, tags in train_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    word_to_ix["未知"] = len(word_to_ix)
    return word_to_ix

def train(model,train_data,word_to_ix,out_filepath,model_savepath,tag_to_ix,ix_to_tag):
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    fp = open(out_filepath,"w+",encoding="utf-8")
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(
            EPOCH): 
        se_count = 0
        all_acc = 0
        for sentence, tags in train_data:
            se_count += 1
            model.zero_grad()      
            ans = [tag_to_ix[i] for i in tags]
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)     
            sentence_in = sentence_in.to(device)
            targets = targets.to(device)
            
            loss = model.neg_log_likelihood(sentence_in, targets)
            output,output_tags = model(sentence_in)
            
            for i in range(len(sentence)):
                out = sentence[i] + " " + ix_to_tag[output_tags[i]]
                fp.write(out)
                fp.write('\n')
            fp.write('\n')
            
            ans = np.array(ans)
            output_tags = np.array(output_tags)
            correct = sum(ans == output_tags)
            accuracy = correct / len(ans)
            all_acc += accuracy
            print("{}/{}  accuracy:{}".format(se_count,train_data.__len__(),accuracy))


            loss.backward()
            optimizer.step()

    torch.save(model,model_savepath)
    print("Result Acc:{}".format( all_acc / train_data.__len__() ))
    
def test(model,train_data,word_to_ix,out_filepath,tag_to_ix,ix_to_tag):
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    fp = open(out_filepath,"w+",encoding="utf-8")


    for epoch in range(
            1): 
        se_count = 0
        all_acc = 0
        for sentence, tags in train_data:
            se_count += 1
            
            ans = [tag_to_ix[i] for i in tags]


            sentence_in = prepare_sequence(sentence, word_to_ix)

            
            sentence_in = sentence_in.to(device)

            output,output_tags = model(sentence_in)
            
            for i in range(len(sentence)):
                out = sentence[i] + " " + ix_to_tag[output_tags[i]]
                fp.write(out)
                fp.write('\n')
            fp.write('\n')
            
            ans = np.array(ans)
            output_tags = np.array(output_tags)
            correct = sum(ans == output_tags)
            accuracy = correct / len(ans)
            all_acc += accuracy
            print("{}/{}  accuracy:{}".format(se_count,train_data.__len__(),accuracy))


 
    print("Result Acc:{}".format( all_acc / train_data.__len__() ))
