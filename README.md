# DeepLearning-for-sql-injection
LSTM and MLP for sql injection detection

Detection SQL Injection with Deep Learning
环境
tensorflow

keras

说明
data目录下是使用的数据，包括一个xss样本、sql注入样本和负样本。
file目录保存训练好的词向量、预处理的数据、训练好的模型等。
log目录保存训练日志，可用tensorborad可视化。
RUN
运行 word2vec_ginsim.py训练嵌入式词向量
运行processing.py预处理数据，生成训练数据和测试数据。
MLP.py、LSTM.py、Conv.py分别使用多层感知机、长短时记忆、卷积神经网络训练模型，在测试集上准确率和召回率。
