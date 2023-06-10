from BiLSTM_CRF import *
import check

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 500
HIDDEN_DIM = 400
EPOCH = 1

E_tag_to_ix = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC":5, "I-LOC":6, "B-MISC":7,"I-MISC":8, START_TAG:9, STOP_TAG:10}
E_ix_to_tag = {0:"O", 1:"B-PER", 2:"I-PER", 3:"B-ORG", 4:"I-ORG", 5:"B-LOC", 6:"I-LOC", 7:"B-MISC", 8:"I-MISC", 9:START_TAG, 10:STOP_TAG}


C_tag_to_ix = {"O":0, "B-NAME":1, "M-NAME":2, "E-NAME":3, "S-NAME":4, "B-CONT":5, "M-CONT":6,
               "E-CONT":7, "S-CONT":8, "B-EDU":9, "M-EDU":10, "E-EDU":11, "S-EDU":12, "B-TITLE":13,
               "M-TITLE":14, "E-TITLE":15, "S-TITLE":16, "B-ORG":17, "M-ORG":18, "E-ORG":19, "S-ORG":20,
               "B-RACE":21, "M-RACE":22, "E-RACE":23, "S-RACE":24, "B-PRO":25, "M-PRO":26, "E-PRO":27,
               "S-PRO":28, "B-LOC":29, "M-LOC":30, "E-LOC":31, "S-LOC":32, START_TAG:33, STOP_TAG:34
               }
C_ix_to_tag = {0:"O", 1:"B-NAME", 2:"M-NAME", 3:"E-NAME", 4:"S-NAME", 5:"B-CONT", 6:"M-CONT",
               7:"E-CONT", 8:"S-CONT", 9:"B-EDU", 10:"M-EDU", 11:"E-EDU", 12:"S-EDU", 13:"B-TITLE",
               14:"M-TITLE", 15:"E-TITLE", 16:"S-TITLE", 17:"B-ORG", 18:"M-ORG", 19:"E-ORG", 20:"S-ORG",
               21:"B-RACE", 22:"M-RACE", 23:"E-RACE", 24:"S-RACE", 25:"B-PRO", 26:"M-PRO", 27:"E-PRO",
               28:"S-PRO", 29:"B-LOC", 30:"M-LOC", 31:"E-LOC", 32:"S-LOC", 33:START_TAG, 34: STOP_TAG
                }



C_data_path = "./Part 3/Chinese/train.txt"
C_model_savepath = "./Part 3/model/model7(C)"
C_out_filepath = "./Part 3/output/output(C).txt"
C_test_data_path = "./Part 3/Chinese/validation.txt"
E_data_path = "./Part 3/English/train.txt"
E_test_data_path = "./Part 3/English/validation.txt"
E_output_filepath = "./Part 3/output/output(E).txt"
E_model_savepath = "./Part 3/model/model2(E)"
result_path = "./Part 3/result.txt"



mode = "English"

if mode == "English":
    data_path = E_data_path
    test_data_path = E_test_data_path
    out_filepath = E_output_filepath
    model_savepath = E_model_savepath
    tag_to_ix = E_tag_to_ix
    ix_to_tag = E_ix_to_tag
else:
    data_path = C_data_path
    test_data_path = C_test_data_path
    out_filepath = C_out_filepath
    model_savepath = C_model_savepath
    tag_to_ix = C_tag_to_ix
    ix_to_tag = C_ix_to_tag

train_data =  get_data(data_path)
word_ix = get_word_ix(train_data)
test_data = get_data(test_data_path)
# word_ix = get_word_ix(test_data)
model = torch.load(model_savepath)
# model = BiLSTM_CRF(len(word_ix), E_tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
# train(model,train_data,word_ix,out_filepath,model_savepath,C_tag_to_ix,C_ix_to_tag)
# test(model,test_data,word_ix,out_filepath,tag_to_ix,ix_to_tag)

fp = open(out_filepath,'r+',encoding="utf-8")
fp2 = open(result_path,'w+',encoding="utf-8")

lines = fp.readlines()
n = lines.__len__()
for i in range(n):
    if lines[i] != '\n':
        word , pred = lines[i].rstrip().split()
        fp2.write(word + " " + pred + '\n')
    else:
        if i == n - 1:
            continue
        else:
            fp2.write('\n')
fp.close()
fp2.close()

check.check(mode,test_data_path,out_filepath)