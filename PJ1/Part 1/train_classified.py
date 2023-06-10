import BP_classified as bp
# import ver2 as bp
import numpy as np
import data
import pickle


dataset , ansSet = data.getData_ver2()

Act = 'sigmod'
# Ne = bp.CreateNetwork((9*9,150,225,300,128,64,12),Act)
Ne = bp.CreateNetwork((9*9,128,256,128,64,32,24,12),Act)
Ne.train(dataset,ansSet,0.015,10000,64,0.4)
Accuracy = Ne.test(dataset,ansSet,10000)
print("对测试集的准确率为：{}%".format(Accuracy * 100))
with open("Ne2.pickle","wb") as f:
    pickle.dump(Ne,f)
