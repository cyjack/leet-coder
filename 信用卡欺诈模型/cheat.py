import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr#引入数值计算包的统计函数
import joblib
from collections import Counter

from imblearn.over_sampling import RandomOverSampler #过采样
from sklearn.svm import SVC


def read_data(path):
    data = pd.read_csv('creditcard.csv')
    #读取数据

    name = data.columns.tolist()
    #定义每个特征名称

    data  = np.array(data)
    #将数据转化成numpy的格式

    x = np.array([data[i][0:29] for i in range(len(data))])
    x = standard(x)

    #特征的分离及标准化
    y = np.array([data[i][-1:] for i in range(len(data))]).reshape(len(x),)
    #分类标签的分离
    return  x,y,name

def standard(x):
    ss = StandardScaler()
    x  = ss.fit_transform(x)
    #标准化
    return x

def sigle_value(x,y,name):
    rf = RandomForestClassifier(n_estimators=20, max_depth=4)
    #初始化一个svm值


    scores =[]
    #保存最终的评价结果

    for i in trange(x.shape[1]):
        #有30个维度，每个维度计算预测标签的分类的结果

        score = cross_val_score(rf,x[:,i:i+1],y,scoring = 'f1',
                                cv = ShuffleSplit(5,test_size=0.2))
        #使用五折交叉检验，调和平均f1作为衡量标准，训练集和测试集的比例为8:2

        scores.append((round(np.mean(score),3),name[i]))
        #然后每训练完一个模型，就对每个模型输出的f1进行求平均，留下后面的三位小数。

    print(sorted(scores,reverse = True))#输出特征的排序
    #输出对每个维度衡量的结果排序

def multi_value(x,y,name):
    '''
    """
使用RFE进行特征选择：RFE是常见的特征选择方法，也叫递归特征消除。它的工作原理是递归删除特征，
并在剩余的特征上构建模型。它使用模型准确率来判断哪些特征（或特征组合）对预测结果贡献较大。
"""
    :param x:
    :param y:
    :param name:
    :return:
    '''
    rf = RandomForestClassifier(n_estimators=20, max_depth=5)

    #初始化一个svm值
    rfe = RFE(rf,n_features_to_select=1,step = 1)
    #最终选择5个特征进行分类，每次去掉一个特征
    rfe.fit(x,y)
    #训练

    print("Features sorted by their rank:")
    print(sorted(zip(map(lambda x: round(x, 4),rfe.ranking_), name)))
    #输出特征的排序

def rf(x,y,name):
    rf = RandomForestClassifier(n_estimators=200, max_depth=4)
    #初始化一个svm值
    ros = RandomOverSampler(random_state= 0)
    #初始化过采样样本
    x_resample,y_resample = ros.fit_sample(x,y)
    score = cross_val_score(rf, x_resample, y_resample, scoring='f1',
                    cv=ShuffleSplit(5, test_size=0.2))
    #五折交叉检验评价模型
    print(np.mean(score))

def svm(x,y,name):
    svm = SVC(C=10, kernel='rbf', degree=3,max_iter=10000)
    #初始化一个svm值
    ros = RandomOverSampler(random_state= 0)
    #初始化过采样样本
    x_resample,y_resample = ros.fit_sample(x,y)

    # print(sorted(Counter(y_resample).items()))#排序计算的结果
    svm.fit(x_resample,y_resample)
    #训练svm
    joblib.dump(svm, 'svm')
    #存储模型
    score = cross_val_score(svm, x_resample, y_resample, scoring='f1',
                    cv=ShuffleSplit(5, test_size=0.2))
    #五折交叉检验评价模型
    print(np.mean(score))


def pearsnor_cal(x,y,name):

   p_all = []
   for i in range(x.shape[1]):

        p_all.append((round(abs(pearsonr(x[:, i], y)[0]),3),name[i]))
        #计算皮尔逊相关系数取绝对值


   print(sorted(p_all, reverse=True))
   #输出排序结果

def svm_predict(x):
    svm = joblib.load('svm')
    y = svm.predict(x)
    print(y)



# def predict(x):
x,y,name =read_data('creditcard.csv')



# sigle_value(x,y,name)
#单一变量特征选择
# multi_value(x,y,name)
#多变量递归删除特征选择

# rf(x,y,name)
# #训练基于随机森林的分类模型

svm(x,y,name)
#基于支持向量机的分类模型


# pearsnor_cal(x,y,name)
# #计算皮尔逊相关系数

# svm_predict(x)
##预测结果

