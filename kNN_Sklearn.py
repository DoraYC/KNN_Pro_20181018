
# coding: utf-8

# In[8]:


# -*- coding: UTF-8 -*-

import numpy as np
import operator
import time
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
函数说明：将 32x32的二进制图像转换为1x1024向量

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量
    
"""

def img2vector(filename):
    #创建1x1024零向量
    returnVect = np.zeros((1,1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        lineStr = fr.readline()
        #每行前32个元素依次添加到returnVect 中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect

"""
函数说明： Sklearn 手写数字分类测试

Parameters:
    无
Returns:
    无
"""
def handwritingClassTest():
    #测试集的Labels
    hwLabels = []
    trainingFileList = listdir('./digits/trainingDigits')
    m = len(trainingFileList)
    #初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector('./digits/trainingDigits/%s' % (fileNameStr))
    #构建kNN分类器 n_neighbors 为kNN的k值，即选取最近的k个点，algorithm搜索算法为默认
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    #拟合模型， trainingMat 为测试矩阵，hwLabels 为对应的标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits 目录下的文件列表
    testFileList = listdir('./digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量，用于训练
        vectorUnderTest = img2vector('./digits/testDigits/%s' % (fileNameStr))
        #predict() 获得预测结果
        #classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels,3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))

    
"""
函数说明：main 函数

Parameters:
    无
    
Returns:
    无
"""
if __name__ == '__main__':
    start = time.clock()
    handwritingClassTest()
    end = time.clock()
    print("程序运行时间：%s 秒" % (end - start))
            
        
        
        
        

