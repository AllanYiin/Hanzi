import _pickle as cPickle
import gzip
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class word2vec(dict):
    def __init__(self, filename='../Data/word2vec.pklz'):

        """
        Py Word2vec结构
        """
        super().__init__()
        self.name = 'WordEmbedding'
        self.load(filename)
        self.vocab_cnt = len(self)
        self.dims = self[list(self.keys())[0]].shape[0]
        print('詞彙數:' + str(self.vocab_cnt))
        print('維度數:' + str(self.dims))
        self.word2idx = {w: i for i, w in enumerate(self.keys())}
        self.idx2word = {i: w for i, w in enumerate(self.keys())}
        self._matrix = np.array(list(self.values()))
        print(self._matrix.shape)

    def save(self, filename='word2vec.pklz'):
        """
        :param filename:存储位置
        """
        fil = gzip.open(filename, 'wb')
        cPickle.dump(self, fil, protocol=pickle.HIGHEST_PROTOCOL)
        fil.close()

    def load(self, filename='word2vec.pklz'):
        fil = gzip.open(filename, 'rb')
        while True:
            try:
                tmp = cPickle.load(fil)
                self.update(tmp)
            except EOFError as e:
                print(e)
                break
        fil.close()

    def get_all_vocabs(self):
        '''
        回傳所有詞彙清單
        :return:
        '''
        return list(self.keys())

    def cosine_distance(self, representation1, representation2, axis=-1):
        """
        計算兩個表徵間的cosine距離
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
            print('status3')
            product = array1 * array2
            print(product.shape)
            return np.sum(product, -1) / (array1.shape[-1] * array2.shape[-1])

    def euclidean_distance(self, representation1, representation2, axis=-1):
        """

        計算兩個表徵間的歐幾里得距離
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

    def manhattan_distance(self, representation1, representation2, axis=-1):
        """

        計算兩個表徵間的歐幾里得距離
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

    # 找尋空間最近字
    def find_nearest_word(self, represent, topk: int = 10, stopwords: list = []):

        """
        根據表徵(可以是字，可以是詞向量)取得最接近的詞
        :param stopwords: 停用詞，將此類詞排除於答案
        :param represent:
        :param topk:
         :param 什麼距離公式
        :return:

        """

        array1 = np.empty(200)
        if isinstance(represent, str) and represent in self:
            array1 = self[represent]
            stopwords.append(represent)  # 排除原輸入字成為輸入

        elif isinstance(represent, np.ndarray):
            array1 = represent
        else:
            raise NotImplementedError

        result_cos = cosine_similarity(np.reshape(array1, (1, array1.shape[-1])), self._matrix)
        result_cos = np.reshape(result_cos, result_cos.shape[-1])

        result_sort = result_cos.argsort()[-1 * topk:][::-1]
        result = []
        for idx in result_sort:
            if self.idx2word[idx] not in stopwords and sum(
                    [1 if stop.startswith(self.idx2word[idx]) else 0 for stop in stopwords]) == 0:
                if self.idx2word[idx] != stopwords[0] or len(stopwords) == 0:
                    result.append((self.idx2word[idx], result_cos[idx]))
            return result

    # 類比關係
    def analogy(self, wordA: str, wordB: str, wordC: str, topk: int = 10, stopwords: list = []):

        """
        語意類比關係  A:B=C:D
        :param wordA:
        :param wordB:
        :param wordC:
        :param topk: 取前K個
        :return:
        """

        if wordA in self and wordB in self and wordC in self:
            arrayD = self[wordB] - self[wordA] + self[wordC]
            result_cos = cosine_similarity(np.reshape(arrayD, (1, arrayD.shape[-1])), self._matrix)
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
                    if self.idx2word[idx] != wordA or self.idx2word[idx] != wordB:
                        result.append((self.idx2word[idx], result_cos[idx]))
            return result
        else:
            return None

    # 根據兩個案例產生枚舉清單
    def get_enumerator(self, wordA: str, wordB, topk: int = 20):
        centroid = (self[wordA] + self[wordB]) / 2
        result_cos = cosine_similarity(np.reshape(centroid, (1, centroid.shape[-1])), self._matrix)
        result_cos = np.reshape(result_cos, result_cos.shape[-1])
        result_sort = result_cos.argsort()[-1 * topk:][::-1]
        result = []
        stopwords = []
        stopwords.append(wordA)
        stopwords.append(wordB)
        for idx in result_sort:
            if self.idx2word[idx] not in stopwords and sum(
                    [1 if stop.startswith(self.idx2word[idx]) else 0 for stop in stopwords]) == 0:
                if self.idx2word[idx] != wordA and self.idx2word[idx] != wordB:
                    result.append((self.idx2word[idx], result_cos[idx]))
        return result

    # 根據三個案例產生枚舉清單
    def get_enumerator1(self, wordA: str, wordB: str, wordC: str, topk: int = 20):
        # 對，別懷疑真的就是這樣簡單
        centroid = (self[wordA] + self[wordB] + self[wordC]) / 3

        result_cos = cosine_similarity(np.reshape(centroid, (1, centroid.shape[-1])), self._matrix)
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
        return result

        # 根據二個案例加上一個負樣本產生枚舉清單

    def get_enumerator2(self, wordA: str, wordB: str, minusCase: str, topk: int = 20):
        # 對，別懷疑真的就是這樣簡單
        centroid = (self[wordA] + self[wordB] - self[minusCase]) / 3

        result_cos = cosine_similarity(np.reshape(centroid, (1, centroid.shape[-1])), self._matrix)
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
        return result

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

    # 如果is_print直接打印其餘則回傳字段
    def print_word_statistics(self, result_list: list, is_print: bool = True):
        if is_print:
            print('、'.join(['{:s}:({:.{prec}f}%)'.format(item[0], item[1] * 100.0, prec=3) for item in result_list]))
        else:
            return '、'.join(['{:s}:({:.{prec}f}%)'.format(item[0], item[1] * 100.0, prec=3) for item in result_list])
