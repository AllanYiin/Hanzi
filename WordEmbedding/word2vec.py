import _pickle as cPickle
import datetime
import gzip
import pickle

import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


# 如果is_print直接打印其余则回传字段
def print_word_statistics(result_list: list, is_print: bool = True, dense_layout=False, prefix=''):
    if len(result_list) > 0:
        if isinstance(result_list[0], tuple):
            if dense_layout:
                if is_print:
                    print('、'.join(
                        [prefix + '{:s}:({:.{prec}f}%)'.format(item[0], item[1] * 100.0, prec=3) for item in
                         result_list]))
                else:
                    return '、'.join(
                        [prefix + '{:s}:({:.{prec}f}%)'.format(item[0], item[1] * 100.0, prec=3) for item in
                         result_list])
            else:
                if is_print:
                    print('\n'.join(
                        [prefix + '{:s}:({:.{prec}f}%)'.format(item[0], item[1] * 100.0, prec=3) for item in
                         result_list]))
                else:
                    return '\n'.join(
                        [prefix + '{:s}:({:.{prec}f}%)'.format(item[0], item[1] * 100.0, prec=3) for item in
                         result_list])

        elif isinstance(result_list[0], list) and isinstance(result_list[0][0], tuple):
            results = []
            for item in result_list:
                if is_print:
                    for subitem in item:
                        print(subitem[2] + ":")
                        break
                    print_word_statistics(item, is_print=True, dense_layout=dense_layout, prefix='     ')
                else:
                    for subitem in item:
                        results.append(subitem[2] + ":")
                        break
                    results.append(
                        print_word_statistics(item, is_print=False, dense_layout=dense_layout, prefix='     '))
            return '\n'.join(results)

        return '无回传结果'








class word2vec(dict):
    def __init__(self, filename='../Data/word2vec.pklz'):

        """
        Py Word2vec结构
        """
        super().__init__()
        self.name = 'WordEmbedding'
        self.filename = filename
        self.load(filename)
        self.vocab_cnt = len(self)
        self.dims = self[list(self.keys())[0]].shape[0]
        print('词汇数:' + str(self.vocab_cnt))
        print('维度数:' + str(self.dims))
        self.word2idx = {w: i for i, w in enumerate(self.keys())}
        self.idx2word = {i: w for i, w in enumerate(self.keys())}
        self._matrix = np.array(list(self.values()))
        print(self._matrix.shape)

    def save(self, filename=None):
        """
        :param filename:存储位置
        """
        if filename == None:
            filename = self.filename
        fil = gzip.open(filename, 'wb')
        cPickle.dump(self, fil, protocol=pickle.HIGHEST_PROTOCOL)
        fil.close()

    def load(self, filename=None):
        st = datetime.datetime.now()

        if filename == None:
            filename = self.filename
        fil = gzip.open(filename, 'rb')
        while True:
            try:
                tmp = cPickle.load(fil)
                self.update(tmp)
            except EOFError as e:
                print(e)
                break
        fil.close()
        et = datetime.datetime.now()
        print('total loading time:{0}'.format(et - st))

    def cosine_distance(self, representation1, representation2, axis=-1):
        """
        计算两个表征间的cosine距离
        :param representation1:
        :param representation2:
        :param axis:
        :return:
        """
        array1 = None
        array2 = None
        if isinstance(representation1, np.ndarray):
            array1 = representation1
        elif isinstance(representation1, np.str):
            array1 = self[representation1]
        else:
            raise NotImplementedError

        if isinstance(representation2, np.ndarray):
            array2 = representation2
        elif isinstance(representation2, np.str):
            array2 = self[representation2]
        else:
            raise NotImplementedError
        if len(array1.shape) == 1 and len(array2.shape) == 1:
            return np.sum(array1 * array2, axis) / (len(array1) * len(array2))
        else:
            return

    def euclidean_distance(self, representation1, representation2, axis=-1):
        """

        计算两个表征间的欧几里得距离
        :param representation1:
        :param representation2:
        :param axis:
        :return:
        """
        array1 = None
        array2 = None
        if isinstance(representation1, np.ndarray):
            array1 = representation1
        elif isinstance(representation1, np.str):
            array1 = self[representation1]
        else:
            raise NotImplementedError

        if isinstance(representation2, np.ndarray):
            array2 = representation2
        elif isinstance(representation2, np.str):
            array2 = self[representation2]
        else:
            raise NotImplementedError
        return np.sqrt(np.sum(np.square(array1 - array2)))

    def pairmanhattan_distance(self, representation1, representation2, axis=-1):
        """

        计算两个表征间的欧几里得距离
        :param representation1:
        :param representation2:
        :param axis:
        :return:
        """
        array1 = None
        array2 = None
        if isinstance(representation1, np.ndarray):
            array1 = representation1
        elif isinstance(representation1, np.str):
            array1 = self[representation1]
        else:
            raise NotImplementedError

        if isinstance(representation2, np.ndarray):
            array2 = representation2
        elif isinstance(representation2, np.str):
            array2 = self[representation2]
        else:
            raise NotImplementedError
        f = np.dot(array1.sum(axis=1).reshape(-1, 1), array2.sum(axis=1).reshape(-1, 1).T)
        return f / array2.sum(axis=1) - array2.sum(axis=1)

    # similarity都是越接近1为相似
    def cosine_similarity(self, representation1, representation2):
        array1 = None
        array2 = None
        if isinstance(representation1, np.ndarray):
            array1 = representation1
            if len(array1.shape) == 1:
                array1 = np.reshape(array1, (1, array1.shape[-1]))
        elif isinstance(representation1, np.str):
            array1 = self[representation1]
            array1 = np.reshape(array1, (1, array1.shape[-1]))
        else:
            raise NotImplementedError

        if isinstance(representation2, np.ndarray):
            array2 = representation2
            if len(array2.shape) == 1:
                array2 = np.reshape(array1, (2, array2.shape[-1]))
        elif isinstance(representation2, np.str):
            array2 = self[representation2]
            array2 = np.reshape(array2, (1, array2.shape[-1]))
        else:
            raise NotImplementedError

        return sklearn.metrics.pairwise.cosine_similarity(array1, array2)

    def euclidean_similarity(self, representation1, representation2):
        return self.euclidean_distance(representation1, representation2)

    def manhattan_similarity(self, representation1, representation2):
        return self.manhattan_distance(representation1, representation2)

    def get_all_vocabs(self):
        '''
        回传所有词汇清单
        :return:
        '''
        return list(self.keys())

    # 找寻空间最近字
    def find_nearest_word(self, represent, topk: int = 10, stopwords: list = [], similarity=cosine_distance):

        """
        根据表征(可以是字，可以是词向量)取得最接近的词
        :param stopwords: 停用词，将此类词排除于答案
        :param represent:
        :param topk:
        :return:

        """
        represent_str = ''
        array1 = np.empty(200)
        if isinstance(represent, str) and represent in self:
            represent_str = represent
            array1 = self[represent]
            array1 = np.reshape(array1, (1, array1.shape[-1]))
            stopwords.append(represent)  # 排除原输入字成为输入

        elif isinstance(represent, np.ndarray):
            array1 = represent
            if len(array1.shape) == 1:
                array1 = np.reshape(array1, (1, array1.shape[-1]))
        else:
            raise NotImplementedError
        # array1  (200)=>(1,200)去和整个矩阵进行距离计算
        result_cos = cosine_similarity(array1, self._matrix)
        result_cos = np.reshape(result_cos, result_cos.shape[-1])

        result_sort = result_cos.argsort()[-1 * topk * 3:][::-1]  #先預留3倍 好扣除stopwords
        result = []
        for idx in result_sort:
            if self.idx2word[idx] not in stopwords and sum(
                    [1 if stop.startswith(self.idx2word[idx]) else 0 for stop in stopwords]) == 0:
                if (len(stopwords) > 0 and self.idx2word[idx] != stopwords[0]) or len(stopwords) == 0:
                    result.append((self.idx2word[idx], result_cos[idx], represent_str))
        return result[:min(topk, len(result))]

    # 模拟关系
    def analogy(self, wordA, wordB, wordC, topk: int = 10, stopwords: list = []):

        """
        语意模拟关系  A:B=C:D
        :param wordA:
        :param wordB:
        :param wordC:
        :param topk: 取前K个
        :return:
        """
        arrayA = []
        arrayB = []
        arrayC = []
        if isinstance(wordA, str) and wordA in self:
            arrayA = np.asarray(self[wordA])
            stopwords.append(wordA)
        elif isinstance(wordA, np.ndarray):
            arrayA = wordA
        if isinstance(wordB, str) and wordB in self:
            arrayB = np.asarray(self[wordB])
            stopwords.append(wordB)
        elif isinstance(wordB, np.ndarray):
            arrayB = wordB
        if isinstance(wordC, str) and wordC in self:
            arrayC = np.asarray(self[wordC])
            stopwords.append(wordC)
        elif isinstance(wordC, np.ndarray):
            arrayC = wordC
        if isinstance(wordC, str) or isinstance(wordC, np.ndarray):
            arrayD = arrayB - arrayA + arrayC
            arrayD = np.reshape(arrayD, (1, arrayD.shape[-1]))
            result_cos = cosine_similarity(arrayD, self._matrix)
            result_cos = np.reshape(result_cos, result_cos.shape[-1])
            result_sort = result_cos.argsort()[-1 * topk * 3:][::-1]
            result = []

            for idx in result_sort:
                if self.idx2word[idx] not in stopwords and sum(
                        [1 if stop.startswith(self.idx2word[idx]) else 0 for stop in stopwords]) == 0:
                    result.append((self.idx2word[idx], result_cos[idx], wordC))
            return result[:min(topk, len(result))]

        elif isinstance(wordC, list):
            results = []
            for c in wordC:
                if isinstance(c, str):
                    stopwords.extend(wordC)
                    r = self.analogy(wordA, wordB, c, topk, stopwords)
                    results.append(r)
                elif isinstance(c, tuple) and len(c) >= 2 and isinstance(c[0], str):
                    stopwords.extend([c[0] for c in wordC])
                    r = self.analogy(wordA, wordB, c[0], topk, stopwords)
                    results.append(r)
            return results
        else:
            return None

    # 根据两个案例产生枚举清单
    def get_enumerator(self, wordA: str, wordB, topk: int = 20):
        '''

        :param wordA:
        :param wordB:
        :param topk:
        :return: 回传tuple清单(词,相似度)list of tuple(word,silimarity)
        '''
        centroid = (self[wordA] + self[wordB]) / 2
        centroid = np.reshape(centroid, (1, centroid.shape[-1]))
        result_cos = self.cosine_similarity(centroid, self._matrix)
        result_cos = np.reshape(result_cos, result_cos.shape[-1])
        result_sort = result_cos.argsort()[-1 * topk * 3:][::-1]
        result = []
        stopwords = []
        stopwords.append(wordA)
        stopwords.append(wordB)
        for idx in result_sort:
            if self.idx2word[idx] not in stopwords and sum(
                    [1 if stop.startswith(self.idx2word[idx]) else 0 for stop in stopwords]) == 0:
                if self.idx2word[idx] != wordA and self.idx2word[idx] != wordB:
                    result.append((self.idx2word[idx], result_cos[idx]))
        return result[:min(topk, len(result))]

    # 根据三个案例产生枚举清单
    def get_enumerator1(self, wordA: str, wordB: str, wordC: str, topk: int = 20):
        # 对，别怀疑真的就是这样简单
        centroid = (self[wordA] + self[wordB] + self[wordC]) / 3
        centroid = np.reshape(centroid, (1, centroid.shape[-1]))
        result_cos = self.cosine_similarity(centroid, self._matrix)
        result_cos = np.reshape(result_cos, result_cos.shape[-1])
        result_sort = result_cos.argsort()[-1 * topk:][::-1]
        result = []
        stopwords = []
        stopwords.append(wordA)
        stopwords.append(wordB)
        stopwords.append(wordC)
        for idx in result_sort:
            if self.idx2word[idx] not in stopwords and sum(
                    [1 if stop.startswith(self.idx2word[idx]) else 0 for stop in stopwords]) == 0:
                if self.idx2word[idx] != wordA and self.idx2word[idx] != wordB:
                    result.append((self.idx2word[idx], result_cos[idx]))
        return result[:min(topk, len(result))]

        # 根据二个案例加上一个负样本产生枚举列表

    def get_enumerator2(self, wordA: str, wordB: str, minusCase: str, topk: int = 20):
        # 对，别怀疑真的就是这样简单
        centroid = (self[wordA] + self[wordB] - self[minusCase]) / 3
        centroid = np.reshape(centroid, (1, centroid.shape[-1]))
        result_cos = self.cosine_similarity(centroid, self._matrix)
        result_cos = np.reshape(result_cos, result_cos.shape[-1])
        result_sort = result_cos.argsort()[-1 * topk:][::-1]
        result = []
        stopwords = []
        stopwords.append(wordA)
        stopwords.append(wordB)
        stopwords.append(minusCase)
        for idx in result_sort:
            if self.idx2word[idx] not in stopwords and sum(
                    [1 if stop.startswith(self.idx2word[idx]) else 0 for stop in stopwords]) == 0:
                if self.idx2word[idx] != wordA or self.idx2word[idx] != wordB:
                    result.append((self.idx2word[idx], result_cos[idx]))
        return result[:min(topk, len(result))]

    def get_antonyms(self, wordA: str, topk: int = 10, ispositive: bool = True):
        seed = [['美丽', '丑陋'], ['安全', '危险'], ['成功', '失败'], ['富有', '贫穷'], ['快乐', '悲伤']]
        proposal = {}
        for pair in seed:
            if ispositive:
                result = self.analogy(pair[0], pair[1], wordA, topk)
                print(self.find_nearest_word((self[pair[0]] + self[pair[1]]) / 2, 3))
            else:
                result = self.analogy(pair[1], pair[0], wordA, topk)
                print(self.find_nearest_word((self[pair[0]] + self[pair[1]]) / 2, 3))
            for item in result:
                term_products = np.argwhere(self[wordA] * self[item[0]] < 0)
                # print(item[0] + ':' +wordA + str(term_products))
                # print(item[0] + ':' +wordA+'('+str(pair)+')  '+ str(len(term_products)))
                if len(term_products) >= self.dims / 4:
                    if item[0] not in proposal:
                        proposal[item[0]] = item[1]
                    elif item[1] > proposal[item[0]]:
                        proposal[item[0]] += item[1]

        for k, v in proposal.items():
            proposal[k] = v / len(seed)
        sortitems = sorted(proposal.items(), key=lambda d: d[1], reverse=True)
        return [sortitems[i] for i in range(min(topk, len(sortitems)))]
