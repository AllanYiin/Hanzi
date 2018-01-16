from WordEmbedding.word2vec import *

if __name__ == '__main__':
    w2v = word2vec()

    # anto = w2v.get_antonyms('尊敬')
    # print(''.join([item[0] + str(w2v.cosine_distance('尊敬', item[0])) for item in anto]))
    # print_word_statistics(anto)

    # print_word_statistics(w2v.find_nearest_word('似乎', 10))

    print('')
    print('')

    # print('双子座之于花心，那么处女座则是…' +'\n'+ print_word_statistics(w2v.analogy('双子座', '花心', '处女座'), False))
    print('####成为深度占星大师')
    stars = ['白羊座', '金牛座', '巨蟹座', '狮子座', '处女座', '天秤座', '天蝎座', '射手座', '摩羯座', '水瓶座',
             '双鱼座']  # [item[0] for item in  w2v.get_enumerator("双子座", "金牛座", 20)]
    # stars.append("双子座")

    print(stars)
    print("模式一:传统类比关系模式")
    print_word_statistics(w2v.analogy('双子座', '花心', stars, 5, ['花心']))

    print("二：多个类比平均")
    print_word_statistics(
        w2v.analogy(np.add(w2v['双子座'], w2v['金牛座']) / 2, np.add(w2v['花心'], w2v['小气']) / 2, stars, 5, ['花心', '小气']))

    print("三：同性质类比概念")
    print_word_statistics(w2v.analogy('双子座', np.add(w2v['花心'], w2v['聪明']) / 2, stars, 5, ['花心', '聪明']))

    print("四：抽象概念增强")
    print_word_statistics(
        w2v.analogy(np.add(w2v['双子座'], w2v['星座']) / 2, np.add(w2v['花心'], w2v['性格']) / 2, stars, 5, ['花心', '性格']))

    print('####列举国家清单然后推估他们的首都')
    answer = w2v.get_enumerator("菲律宾", "泰国", 20)
    cptitals = [item[0] for item in w2v.get_enumerator("北京", "曼谷", 20)]
    print(cptitals)
    # 比较一下这个
    cptitals = [item[0] for item in w2v.get_enumerator2("北京", "曼谷", "菲律宾", 20)]  # 找与北京曼谷相似的，但是要把国家排除
    print(cptitals)

    print('枚举与菲律宾、泰国相似概念的实体…' + print_word_statistics(answer, False))

    capital_country = w2v["华盛顿"] - w2v["美国"]  # 建构首都与国家关系式(可以试试不同建构方式)

    for item in answer:
        country = item[0]
        capital = w2v[item[0]] + capital_country
        for similarword in w2v.find_nearest_word(capital, 5):
            if str(similarword[0]) != country or similarword[0] in cptitals:
                print(country + "=>" + similarword[0])
                break
                # 有没有甚么好办法可以筛选的更精确?
                # 能否设计出其他的透过词向量拓展知识图谱的ˊ类似方法流程
