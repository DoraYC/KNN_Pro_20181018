
# coding: utf-8

# ## 使用k-近邻算法将每组数据划分到某个类中

# In[24]:


import operator 


def classify0(inX, dataSet, labels, K):
    #numpy 函数shape[0]返回dataSet行数
    dataSetSize = dataSet.shape[0]
    #将intX在横向重复dataSetSize次，纵向重复1次
    #例如intX=([1,2])--->([[1,2],[1,2],[1,2],[1,2]])便于后面计算
    #对应差值
    #tile: 在行列方向上重复
    #tile(a,(m,n)) 通过n份a的拷贝创建m, tile(A,n)将数组A重复n次构成新数组
    #tile(a,x):   x是控制a重复几次的，结果是一个一维数组
    #tile(a,(x,y))：   结果是一个二维矩阵，其中行数为x，列数是一维数组a的长度和y的乘积
    #tile(a,(x,y,z)):   结果是一个三维矩阵，其中矩阵的行数为x，矩阵的列数为y，而z表示矩阵每个单元格里a重复的次数,
    #(三维矩阵可以看成一个二维矩阵，每个矩阵的单元格里存者一个一维矩阵a)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat ** 2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，计算出距离
    distances = sqDistances ** 0.5
    #方法argsort()返回distances中元素从小到大排序后的索引值
    sortedDistIndicies = distances.argsort()
    #定义一个记录类别次数的字典
    classCount = {}
    for i in range(K):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        #dict.get(key,default=None),字典的get()方法，返回指定键的值，如果值不在字典中返回默认值
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #sorted()对所有可迭代对象排序，key是用于比较的元素，operator.itemgetter(1)获取排序对象的第1个域，
    # reverse=True 降序，False 默认值：升序
    #Python3中没有iteritems()方法，用items()替换
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]


# ## 将文本记录转换为NumPy的解析程序

# In[1]:


import numpy as np


def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    #len()计算文件行数
    numberOfLines = len(arrayOLines)
    #创建返回的NumPy矩阵：特征矩阵
    returnMat = np.zeros((numberOfLines, 3))
    #返回的特征标签
    classLabelVector = []
    #行索引
    index = 0
    for line in arrayOLines:
        #str.strip(rm) 删除str头和尾指定的字符 rm为空时，默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #每行数据是\t划分的，将每行数据按照\t进行切片划分
        listFromLine = line.split('\t')
        #取出前三列数据存放到returnMat特征矩阵中
        returnMat[index, :] = listFromLine[0: 3]
        #将列表的最后一列转换为int类型并存储到classLabelVector向量中
        #classLabelVector.append(int(listFromLine[-1]))    datingTestSet2.txt 用
        #datingTestSet.txt文档最后一列是英文，需要转换,依据文本中标记的喜欢程度进行分类
        if "datingTestSet2.txt" in filename:
            classLabelVector.append(int(listFromLine[-1])) 
        else:
            if listFromLine[-1] == "didntLike":
                classLabelVector.append(1)
            elif listFromLine[-1] == "smallDoses":
                classLabelVector.append(2)
            else:
                classLabelVector.append(3)

        index += 1
    return returnMat, classLabelVector


# ## 归一化特征值

# In[18]:


import numpy as np

def autoNorm(dataSet):
    #公式： newValue = (oldValue - min)/(max - min)
    #取数据集每列最小特征值
    minVals = dataSet.min(0)
    #取数据集每列最大特征值
    maxVals = dataSet.max(0)
    #取值范围
    ranges = maxVals - minVals
    #零矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #当前值-最小值：tile()函数将变量内容复制成输入矩阵同样大小的矩阵，这是具体特征值相除
    #而对于某些数值处理软件包，/可能意味着矩阵除法，但在numpy库中，矩阵除法需要使用函数 linalg.solve(matA,matB)
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以取值范围
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# ## 分类器针对约会网站的测试代码

# In[4]:


def datingClassTest():
    hoRatio = 0.05
    #首先使用 file2matrix ,autoNorm 函数从文件中读取数据并转换为归一化特征值
    datingDataMat, datingLabels = file2matrix('./dating/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #计算测试向量的数量，此步决定了normMat 向量中哪些数据用于测试，哪些用于训练样本
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #将两部分数据输入到分类器函数中
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs: m, :], datingLabels[numTestVecs:m],3)
        print ("The classfier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        #函数计算错误率
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
        print("The total error rate is: %f" % (errorCount / float(numTestVecs)))


# In[5]:


import numpy as np


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    #python 3.0 以后的版本替换 raw_input 为 input
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('./dating/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult -1])
    
    

