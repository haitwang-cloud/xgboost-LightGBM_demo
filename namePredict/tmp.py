# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
import math

# 读取train.txt
train = pd.read_csv('train.txt')
test = pd.read_csv('test.txt')
submit = pd.read_csv('sample_submit.csv')

#把数据分为男女两部分
names_female = train[train['gender'] == 0]
names_male = train[train['gender'] == 1]

totals = {'f': len(names_female),
          'm': len(names_male)}

frequency_list_f = defaultdict(int)
for name in names_female['name']:
    for char in name:
        frequency_list_f[char] += 1. / totals['f']

frequency_list_m = defaultdict(int)
for name in names_male['name']:
    for char in name:
        frequency_list_m[char] += 1. / totals['m']

def LaplaceSmooth(char, frequency_list, total, alpha=1.0):
    count = frequency_list[char] * total
    distinct_chars = len(frequency_list)
    freq_smooth = (count + alpha ) / (total + distinct_chars * alpha)
    return freq_smooth

def GetLogProb(char, frequency_list, total):
    freq_smooth = LaplaceSmooth(char, frequency_list, total)
    return math.log(freq_smooth) - math.log(1 - freq_smooth)

def ComputeLogProb(name, bases, totals, frequency_list_m, frequency_list_f):
    logprob_m = bases['m']
    logprob_f = bases['f']
    for char in name:
        logprob_m += GetLogProb(char, frequency_list_m, totals['m'])
        logprob_f += GetLogProb(char, frequency_list_f, totals['f'])
    return {'male': logprob_m, 'female': logprob_f}

def GetGender(LogProbs):
    return LogProbs['male'] > LogProbs['female']


base_f = math.log(1 - train['gender'].mean())
base_f += sum([math.log(1 - frequency_list_f[char]) for char in frequency_list_m])

base_m = math.log(train['gender'].mean())
base_m += sum([math.log(1 - frequency_list_m[char]) for char in frequency_list_m])

bases = {'f': base_f, 'm': base_m}


result = []
for name in test['name']:
    LogProbs = ComputeLogProb(name, bases, totals, frequency_list_m, frequency_list_f)
    gender = GetGender(LogProbs)
    result.append(int(gender))

submit['gender'] = result

submit.to_csv('my_NB_prediction12.csv', index=False)