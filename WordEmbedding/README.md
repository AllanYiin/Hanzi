由于现行python的word2vec都还存在中文乱码问题，再加上原有格式使用上不若numpy array般直觉且有效率，而且更符合深度学习的使用习惯，因此我们特地将整个(简体)中文词向量重新转换格式，结构为dict，其中key就是词的中文，而value则是200维的词向量，

词汇总数：1,151,245
<br>維度数：200
<br>形状为：(1151245, 200)

再初始化後，以及自動載入詞向量後，會自動計算

转换过的词向量为了方便传输因此先pickle处理后再进行压缩，我们把扩展名定为*.pklz。重新转换过的word2vec.pklz较大，各位可以至以下路径下载:

http://pan.baidu.com/s/1nuJaB01
<br>https://1drv.ms/u/s!AsqOV38qroofiOCAOe-7BxoBMTRPeXs



由于词数过大，因此为了能够获得较佳的计算效能，建议将numpy设定intel mkl链结，以提升执行效率



