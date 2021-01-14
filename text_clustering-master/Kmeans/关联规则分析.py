# -*- coding: utf-8 -*-

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd


import itertools

min_support = 0.4
min_confidence = 0.3


def loadDataSet(filename):
    """
    加载数据集
    :param filename:
    :return:
    """
    datas = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            datas.append(line.strip().split(','))
            print(datas)
    return datas


def count(items, datas):
    """
    计算items出现在data中的个数
    :param items: list []
    :param datas: list[list] [[]]
    :return:
    """
    num = 0
    for data in datas:
        if set(items).issubset(set(data)):
            num += 1
    return num


def judgeConnect(items1, items2):
    """
    判断两个items是否满足连接步的条件
    :param items1:
    :param items2:
    :return:
    """
    if len(items1) != len(items2):
        return False
    diff = 0
    for item in items1:
        if item not in items2:
            diff += 1
    if diff == 1:
        return True
    else:
        return False


def judgeSame(itemscur, itemsall):
    """
    判断当前的items，是否在之前的集合中出现过
    :param itemscur: list []
    :param itemsall: list[list] [[]]
    :return:
    """
    for items in itemsall:
        if sorted(itemscur) == sorted(items):
            return True
    return False


def getF1(datas):
    """
    得到频繁一项集
    :param datas:
    :return:
    """
    c1 = list(set([item for items in datas for item in items]))
    c1 = [[item] for item in c1]
    f1 = []
    s1 = []  # 支持度
    for c in c1:
        support = count(c, datas) / float(len(datas))
        if support >= min_support:
            f1.append(c)
            s1.append(support)
    return f1, s1


def getFk(fk_1, datas):
    """
    由频繁k-1项集得到频繁k项集
    :param fk_1: list[list] [[]]
    :param datas: list[list] [[]]
    :return:
    """
    # 得到所有的k项集
    ck = []
    for i in range(len(fk_1)):
        for j in range(i + 1, len(fk_1)):
            if judgeConnect(fk_1[i], fk_1[j]):
                items = set(fk_1[i] + fk_1[j])
                if not judgeSame(items, ck):
                    ck.append(list(items))

    # 得到所有的频繁k-1项集
    fk = []
    sk = []
    for items in ck:
        support = count(items, datas) / float(len(datas))
        if support >= min_support:
            fk.append(items)
            sk.append(support)
    return fk, sk


def generate_fk(datas):
    """
    生成频繁项集
    :param datas:
    :return:
    """
    f_s = {}
    f1, s1 = getF1(datas)
    for f, s in zip(f1, s1):
        f_s[frozenset(f)] = s
    fk, sk = getFk(f1, datas)
    while fk:
        for f, s in zip(fk, sk):
            f_s[frozenset(f)] = s
        fk, sk = getFk(fk, datas)
    return f_s


def generate_rule(f_s):
    """
    由频繁项集生成规则
    :param f_s:
    :return:
    """
    rules = []
    for key, value in f_s.items():
        if len(key) >= 2:
            rules.extend(rule(key, f_s, []))
            # print(rules)
    return rules


def rule(items, f_s, cur_rule):
    for item in itertools.combinations(items, 1):
        if items - frozenset(item) in f_s.keys() and \
                f_s[items] / f_s[items - frozenset(item)] >= min_confidence:
            cur_rule.append((str([items - frozenset(item)]), str(item),
                             f_s[items] / float(f_s[items - frozenset(item)])))
            rule(items - frozenset(item), f_s, cur_rule)
    return cur_rule



class wordf :
    def __init__(self,word,freq):
        self.word = word
        self.freq = freq

def find_chinese(file):

    pattern = re.compile(r'[u4e00-u9fa5]')#先compile一个正则表达式
    chinese = re.sub(pattern, '', file)#在使用re.sub进行匹配找到合理的子串
    return chinese

class KmeansClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None):
        """
        加载停用词
        :param stopwords:
        :return:
        """
        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []
    def apriori_keywords(self,corpus_path):
        #apriori方法挖掘频繁项集和关联规则
        corpus = self.preprocess_data(corpus_path)
        for i in range(len(corpus)):

            corpus[i] = corpus[i].strip().split(' ')
            a = []
            for j in corpus[i]:
                a.append(j)
            corpus[i] = a

        f_s = generate_fk(corpus)
        print("频繁项集：{} 个".format(len(f_s)))
        for key, value in f_s.items():
            print("{} : {:.2f}".format(key, value))
        rules = generate_rule(f_s)
        print("关联规则：{}个".format(len(rules)))
        for reason, result, conf in rules:
            print("{} ----> {} : {}".format(reason, result, conf))

    
    def preprocess_data(self, corpus_path):
        """
        文本预处理，每行一个文本
        :param corpus_path:
        :return:
        """
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = find_chinese(line)
                line = ' '.join([word for word in jieba.lcut(line.replace(' ','')) if word not in self.stopwords])
                corpus.append(line)
        return corpus


   
    def get_text_tfidf_matrix(self, corpus):
        """
        获取tfidf矩阵
        :param corpus:
        :return:
        """
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))

        # 获取词袋中所有词语
        # words = self.vectorizer.get_feature_names()

        # 获取tfidf矩阵中权重
        weights = tfidf.toarray()
        return weights

    def kmeans(self, corpus_path, n_clusters=5):
        """
        KMeans文本聚类
        :param corpus_path: 语料路径（每行一篇）,文章id从0开始
        :param n_clusters: ：聚类类别数目
        :return: {cluster_id1:[text_id1, text_id2]}
        """
        index = 0
        dict ={}
        with open('../data/class.txt','r',encoding='utf-8') as f:
            for i in f:
                dict.update({str(index):i.strip()})
                index += 1

        print("财经:1","时政:2","娱乐:3","民生:4","房产:5")
        print("文档对应的分类如下")
        print(dict)


        corpus = self.preprocess_data(corpus_path)
        weights = self.get_text_tfidf_matrix(corpus)

        clf = KMeans(n_clusters=n_clusters)

        # clf.fit(weights)

        y = clf.fit_predict(weights)

        # 中心点
        # centers = clf.cluster_centers_

        # 用来评估簇的个数是否合适,距离约小说明簇分得越好,选取临界点的簇的个数
        # score = clf.inertia_

        # 每个样本所属的簇
        result = {}
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result
   






if __name__ == '__main__':

    Kmeans = KmeansClustering(stopwords_path='../data/stop_words.txt')


    Kmeans.apriori_keywords('../data/news')
    #

