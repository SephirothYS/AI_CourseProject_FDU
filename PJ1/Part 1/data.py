import numpy as np
import cv2

def getData():
    x = np.full((28,28), 1 / 255)

    dataset = np.zeros((12,620,28 * 28))
    ansSet = np.zeros((12,620,12))
    for i in range(12):
        for j in range(620):
            filename = "train/%d/%d.bmp" % (i + 1,j + 1)
            img = cv2.imread(filename)
            imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = imGray * x 
            result = result.flatten()
            dataset[i][j] = result
            ansSet[i][j][i] = 1
    
    return dataset,ansSet

def getData_ver2():
    x = np.full((28,28), 1 / 255)

    dataset = np.zeros((12,240,9 * 9))
    ansSet = np.zeros((12,240,12))
    for i in range(12):
        for j in range(240):
            filename = "G:\\test_data\\%d\\%d.bmp" % (i + 1,j + 1)
            img = cv2.imread(filename)
            imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = imGray * x 
            data = np.zeros((9,9))
            row = 0
            col = 0
            for a in range(0,25,3):
                for b in range(0,25,3):
                    data[row][col] = np.sum(result[b:b+3,a:a+3])
                    col += 1
                    if col == 9:
                        col = 0
                        row += 1
            data = data.flatten()
            dataset[i][j] = data
            ansSet[i][j][i] = 1
    
    return dataset,ansSet
