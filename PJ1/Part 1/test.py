import numpy as np
import pickle
import BP_classified as bp
import data
import cv2

f = open("Ne1.pickle","rb")
Ne = pickle.load(f)


dataset , ansSet = data.getData_ver2()
Accuracy = Ne.test_all(dataset,ansSet)

print("对测试集的准确率为：{}%".format(Accuracy * 100))

# filename = "H:/train/%d/%d.bmp" % (1,1)
# img = cv2.imread(filename)