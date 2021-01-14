# -*- coding: utf-8 -*-

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from apriori import *
import time
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
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

    def frequence_key_word(self,corpus_path):

        all =[]
        corpus = self.preprocess_data(corpus_path)
        xx = 0
        for item in corpus:
            print(item)
            L = []
            vectorizer = CountVectorizer()
            #norm不对词频进行归一化，关闭idf进行计算
            transformer = TfidfTransformer(norm = None,use_idf =False)
            tf = transformer.fit_transform(vectorizer.fit_transform([item]))
            word = vectorizer.get_feature_names()

            weight = tf.toarray()


            for i in range(len(weight)):
                for j in range(len(word)):
                    s1 = wordf(word[j],weight[i][j])
                    # print(word[j],weight[i][j])
                    L.append(s1)

            L.sort(key=lambda t:t.freq,reverse=True)

            l = []
            for w in range(10):
                l.append((L[w].word,L[w].freq))
            print(l)
            xx+=1
            self.plot_keyword(l,xx)
            all.append(l)

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


    def get_text_conunt_matrix(self,corpus):
        pass
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
    def plot_keyword(self,L,xx):
            #画出这个文档关键词的内容并保存图
            dict ={}
            for i in L:
                print(i)
                dict.update({i[0]:i[1]})

            s = sorted(dict.items(),key = lambda x:x[1],reverse = False)#直接对字典排序

            x_x = []
            y_y = []
            for i in s:
                x_x.append(i[0])
                y_y.append(i[1])
            fig,ax = plt.subplots()
            x = x_x
            y = y_y
            ax.barh(x,y,color = 'deepskyblue')
            labels = ax.get_xticklabels()

            plt.setp(labels, rotation=0, horizontalalignment='right')
            for a,b in zip(x,y):
                plt.text(b+1,a,b,ha = 'center',va = 'center')
            plt.xlim(0, max(y)+5)
            ax.legend(['label'],loc = 'lower right')
            plt.rcParams['font.sans-serif'] = ['SimHei']#正常显示中文
            plt.ylabel('关键词')
            plt.xlabel('出现次数')
            plt.rcParams['savefig.dpi'] =300
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['figure.figsize'] =(20,8.0)#尺寸
            plt.title('文档出现次数前十的词')
            plt.savefig('../data/keyword/key'+str(xx)+'.png')
            # plt.show()






if __name__ == '__main__':

    Kmeans = KmeansClustering(stopwords_path='../data/stop_words.txt')

    # Kmeans.frequence_key_word('../data/news')
    #
    Kmeans.apriori_keywords('../data/news')
    #

    # result = Kmeans.kmeans('../data/news', n_clusters=5)
    # print("聚类的结果如下")
    # print(result)
