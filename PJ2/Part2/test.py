import os
import check

E_data_path = "./Part2/English/ENGtrain.data"
E_test_path = "./Part2/English/validation.txt"
template_path = "./Part2/template"
E_model_path = "./Part2/model(E)"
C_model_path = "./Part2/model(C)"
E_output_path = "./Part2/output/output(E).txt"
E_result_path = "./Part2/output/result(E).txt"
C_data_path = "./Part2/Chinese/train.txt"
C_test_path = "./Part2/Chinese/validation.txt"
C_output_path = "./Part2/output/output(C).txt"
C_result_path = "./Part2/output/result(C).txt"
MAXEPOCH = "1000"
f = "4"
c = "3.0"
e = "0.00001"

mode = "Chinese"

if mode == "English":
    data_path = E_data_path
    test_path = E_test_path
    output_path = E_output_path
    result_path = E_result_path
    model_path = E_model_path
else:
    data_path = C_data_path
    test_path = C_test_path
    output_path = C_output_path
    result_path = C_result_path
    model_path = C_model_path

# command = "crf_learn -f {} -c {} -m {} -e {} {} {} {} ".format(f,c,MAXEPOCH,e,template_path,data_path,model_path)
# os.system(command)

command = "crf_test -m {} {} > {}".format(model_path,test_path,output_path)
os.system(command)


fp = open(output_path,'r+',encoding="utf-8")
fp2 = open(result_path,'w+',encoding="utf-8")

lines = fp.readlines()
n = lines.__len__()
for i in range(n):
    if lines[i] != '\n':
        word , pre_tag , pred = lines[i].rstrip().split()
        fp2.write(word + " " + pred + '\n')
    else:
        if i == n - 1:
            continue
        else:
            fp2.write('\n')
fp.close()
fp2.close()

check.check(mode,test_path,result_path)