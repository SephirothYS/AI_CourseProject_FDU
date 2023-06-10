
import numpy as np

class HMM():
    
    def __init__(self):
        self.tag2id =  {}
        self.id2tag = {}
        self.word2id = {}
        self.id2word = {}
        self.word_num = 0
        self.tag_num = 0
        self.ini_sta = None
        self.trans = None
        self.emit = None

    def train(self,train_path):
        for line in open(train_path,"r",encoding='utf-8'):
            if line == "\n":
                continue
            items = line.split()
            word, tag = items[0],items[1].rstrip()
            
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)
                self.id2word[len(self.id2word)] = word
            
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)
                self.id2tag[len(self.id2tag)] = tag
                
        self.word_num = len(self.word2id)
        self.tag_num = len(self.tag2id)

        self.ini_sta = np.zeros(self.tag_num)
        self.trans = np.zeros((self.tag_num,self.tag_num))
        self.emit = np.zeros((self.tag_num,self.word_num))

        pre_tag = ""
        for line in open(train_path,'r',encoding='utf-8'):
            if line == "\n": 
                pre_tag = ""
                continue
                
            items = line.split()
            wordId, tagId = self.word2id[items[0]], self.tag2id[items[1].rstrip()]
            
            if pre_tag == "":
                self.ini_sta[tagId] += 1
                self.emit[tagId][wordId] += 1
            else:
                self.emit[tagId][wordId] += 1
                self.trans[self.tag2id[pre_tag]][tagId] += 1
            
            pre_tag = items[1].strip()
            

        self.ini_sta = self.ini_sta/sum(self.ini_sta)
        for i in range(self.tag_num):
            self.trans[i] /= sum(self.trans[i])
            self.emit[i] /= sum(self.emit[i])
        
    def log(self,v):
        if v==0:
            return np.log(v+0.00001)
        return np.log(v)


    def viterbi_decode(self,x):
        
        for i in range(len(x)):
            word = x[i]
            if self.word2id.get(word) == None:
                x[i] = 2
            else:
                x[i] = self.word2id[word]
        x_len = len(x)
        
        dp = np.zeros((x_len,self.tag_num)) 
        index = np.zeros((x_len,self.tag_num),dtype=int) 
        
        for j in range(self.tag_num):
            dp[0][j] = self.log(self.ini_sta[j]) + self.log(self.emit[j][x[0]])  
        

        for i in range(1, x_len):
            for j in range(self.tag_num):
                dp[i][j] = -99999
                for k in range(self.tag_num):
                    score = dp[i-1][k] + self.log(self.trans[k][j]) + self.log(self.emit[j][x[i]])
                    if score > dp[i][j]:   
                        dp[i][j] = score
                        index[i][j] = k  
        
        best_seq = [0]*x_len
        best_seq[x_len-1] = np.argmax(dp[x_len-1])
        
        for i in range(x_len-2, -1, -1):
            best_seq[i] = index[i+1][best_seq[i+1]]
        
        syms = []
        
        for i in range(len(best_seq)):
            syms.append(self.id2tag[best_seq[i]])
        return syms


    def predict(self,test_path,output_path):
        fp = open(test_path,"r",encoding='utf-8')
        fp2 = open(output_path,'w+',encoding='utf-8')
        words = []
        syms = []
        all_accy = 0
        line_count = 0
        for line in fp:
            if line == "\n":
                pred = self.viterbi_decode(words)
                count = 0
                for i in range(len(syms)):
                    if syms[i] == pred[i]: count += 1
                    fp2.write(self.id2word[words[i]] + " " + pred[i] + '\n')
                fp2.write('\n')
                accy = count / len(syms)
                syms.clear()
                words.clear()
                line_count += 1
                print(line_count)
                all_accy += accy
                continue
            items = line.split()
            word, sym = items[0], items[1].rstrip()
            words.append(word)
            syms.append(sym)
        if words.__len__() != 0:
                pred = self.viterbi_decode(words)
                for i in range(len(syms)):
                    fp2.write(self.id2word[words[i]] + " " + pred[i] + '\n')
        fp.close()
        fp2.close()

        print("平均正确率:{}".format( all_accy / line_count))

