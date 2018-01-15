from WordEmbedded.word2vec import *

if __name__ == '__main__':
    w2v = word2vec()

    # anto = w2v.get_antonyms('尊敬')
    # print(''.join([item[0] + str(w2v.cosine_distance('尊敬', item[0])) for item in anto]))
    # w2v.print_word_statistics(anto)
    similarWords = w2v.find_nearest_word('似乎', 10)
    w2v.print_word_statistics(similarWords)

    ####列舉國家清單然後推估他們的首都
    answer = w2v.get_enumerator("菲律宾", "泰国", 20)
    cptitals = [item[0] for item in w2v.get_enumerator("北京", "曼谷", 20)]
    print(cptitals)
    # 比較一下這個
    cptitals = [item[0] for item in w2v.get_enumerator2("北京", "曼谷", "菲律宾", 20)]  # 找與北京曼谷相似的，但是要把國家排除
    print(cptitals)

    print('枚举与菲律宾、泰国相似概念的实体…' + w2v.print_word_statistics(answer, False))

    capital_country = w2v["华盛顿"] - w2v["美国"]  # 建構首都與國家關係式(可以試試不同建構方式)

    for item in answer:
        country = item[0]
        capital = w2v[item[0]] + capital_country
        for similarword in w2v.find_nearest_word(capital, 5):
            if str(similarword[0]) != country or similarword[0] in cptitals:
                print(country + "=>" + similarword[0])
                break
    # 有沒有甚麼好辦法可以篩選的更精確?
    # 能否設計出其他的透過詞向量拓展知識圖譜的ˊ類似方法流程

    print('')
    print('')

    print('双子座之于花心，那么处女座则是…' + w2v.print_word_statistics(w2v.analogy('双子座', '花心', '处女座'), False))
    print('')
    print('')
    staranswer = w2v.get_enumerator("双子座", "处女座", 12)
    stars = [item[0] for item in staranswer]
    print('枚举与双子座、处女座相似概念的实体…' + w2v.print_word_statistics(staranswer, False))
