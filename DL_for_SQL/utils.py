#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'wf'

import nltk
import re
from urllib.parse import unquote
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def GeneSeg(payload):
    payload=payload.lower()#变小写
    payload=unquote(unquote(payload))#解码
    payload,num=re.subn(r'\d+',"0",payload)#数字泛化为"0"
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)
def init_session():
    #gpu_options=tf.GPUOptions(allow_growth=True)
    ktf.set_session(tf.Session())#创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成。