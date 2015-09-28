import numpy as np
import string
import pandas as pd
from collections import Counter

path='/Users/linjunli/Desktop/Natural Language Processing/homework1/'
train = open(path+'brown-train.txt', 'r+')
test=open(path+'brown-test.txt', 'r+')


def test_set(s):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in s if ch not in exclude)

train1=test_set(train.read())
test1=test_set(test.read())

#1. Pad traning and test porpora with <s> and </s>
train1='<s> '+train1.replace("\n", " </s> \n <s> ")[:-7]
test1='<s> '+test1.replace("\n", " </s> \n <s> ")[:-7]

#2. Lowercase all the words in the training and testing datasets
train1=train1.lower()
test1=test1.lower()

#3.1 Replace all words occuring in training data once with the token <unk>
t=test1[:100]
wordcount = pd.DataFrame(Counter(train1.split()).items())
s=list(wordcount[wordcount.ix[:][1]==1].ix[:][0])

train1=train1.split(' ')
train2=[]
for i in train1: 
    if (i in s):train2.append('<unk>')
    else: train2.append(i)

train2=' '.join(words for words in train2)
train2=train2.replace('  ',' ')

#3.2 replace all words in test data not seen in traning with <unk>
WinT=list(np.unique(np.array(train2.split(' '))))
test1=test1.replace('  ',' ').split(' ')
test2=[]
for i in test1:
    if (i not in WinT):test2.append('<unk>')
    else: test2.append(i)
test2=' '.join(words for words in test2)

#Training the models
#Derive the unigram maximum likelihood model
train2split=train2.split()
counts = pd.DataFrame(Counter(train2split).items())
count=counts.copy()
Ncounts=len(counts)
counts.ix[:][1]=counts.ix[:][1]/1.0/Ncounts
counts.columns=['word','prob']

#Derive the bigram maximum likelihood model
from itertools import izip
pairs=[' '.join(pair) for pair in izip(train2split[:-1], train2split[1:])]
countpairs = pd.DataFrame(Counter(pairs).items())
conditional_p=[]
ct2s=Counter(train2split)#gives the frequency of the second word
for i in countpairs.ix[:][0]:
    conditional_p.append([i,ct2s[i.split()[1]]])
conditional_p=pd.DataFrame(conditional_p)
bigrammodel=countpairs.merge(conditional_p,how='inner', on=[0])
bigrammodel['con_p']=map(lambda x,y:x*1.0/y,bigrammodel['1_x'],bigrammodel['1_y'])

# A bigram model with Add-One smoothing
V=len(np.unique(np.array(train2split)))#=14883, N=434804
bigrammodel['AddOne_con_p']=map(lambda x,y:(x+1)*1.0/(y+V),bigrammodel['1_x'],bigrammodel['1_y'])

# A bigram model with discounting and Katz backoff.
#a=1-sum(map(lambda x,y:(x-0.5)*1.0/y,bigrammodel['1_x'],bigrammodel['1_y']))
bigrammodel['FirstWord']=map(lambda x:x.split()[0],bigrammodel[:][0])
groups=bigrammodel.groupby('FirstWord')
wordN=groups.sum().reset_index(drop=False).ix[:,0:2]
bigrammodel=bigrammodel.merge(wordN,how='inner',on='FirstWord')
bigrammodel.columns=['pairs','PairC','LastWordC','con_p','AddOne_con_p','FirstWord','FirstWordC']
bigrammodel['Dsic_con_p']=map(lambda x,y: (x-0.5)/y,bigrammodel['PairC'],bigrammodel['FirstWordC'])

groups2=bigrammodel.groupby('FirstWord')
wordN2=groups2.sum().reset_index(drop=False).ix[:,[0,6]]
bigrammodel=bigrammodel.merge(wordN2,how='inner',on='FirstWord')
bigrammodel.columns=['pairs','PairC','LastWordC','con_p','AddOne_con_p','FirstWord','FirstWordC','Disc_con_p','alpha']
bigrammodel['alpha1']=map(lambda x:1-x,bigrammodel['alpha'])
alpha_df=bigrammodel.ix[:,[5,8]].groupby('FirstWord').mean().reset_index(drop=False)
del bigrammodel['alpha'],bigrammodel['alpha1']
#conditional probability for unobserved pairs: counts.ix[i,1]*alpha_df[i-1,1]

# The percentage of word tokens and word types of the test copora didn't show up in 
# training data 
#(with unk)
N_test=len(test2.split())
V_test=len(np.unique(np.array(test2.split())))
N=len(train2split)
N_ratio=(N-N_test)*1.0/N
#0.9636318893110459
V_ratio=(V-V_test)*1.0/V
#0.8167036215816703

#Compute the probability of following sentences
corpus1='He was laughed off the screen.'[:-1].lower().split()
corpus2='There was no compulsion behind them.'[:-1].lower().split()
corpus3='I look forward to hearing your reply.'[:-1].lower().split()

# Method 1: Unigram maximum likelihood model
prob1=1.0
for item in corpus1:
    prob1=prob1* float(counts[counts['word']==item]['prob'])
#prob1=1.804403265049493e-08
#Used exp(sum(log)) since 'compulsion' never happens in the training data.
prob2=0.0
for item in corpus2:
    short=counts[counts['word']==item]['prob']
    if (len(short)>0):
        prob2=prob2+np.log(float(short))
prob2=np.exp(prob2)
#prob2=1.1920566510317309e-06
prob3=1.0
for item in corpus3:
    prob3=prob3* float(counts[counts['word']==item]['prob'])
#prob3=6.807925774156419e-13

# Method 2: Bigram maximum likelihood model
pairs1=[' '.join(pair) for pair in izip(corpus1[:-1], corpus1[1:])]
pairs2=[' '.join(pair) for pair in izip(corpus2[:-1], corpus2[1:])]
pairs3=[' '.join(pair) for pair in izip(corpus3[:-1], corpus3[1:])]
#prob1=prob2=prob3=0, since some bigrams are non-existent in training data
prob1=0.0
for item in pairs1:
    short=bigrammodel[bigrammodel['pairs']==item]['con_p']
    if (len(short)>0):
        prob1=prob1+np.log(float(short))
prob1=np.exp(prob1)*float(counts[counts['word']==pairs1[0].split()[0]]['prob'])
 #3.2724373802806885e-05

prob2=0.0
for item in pairs2:
    short=bigrammodel[bigrammodel['pairs']==item]['con_p']
    if (len(short)>0):
        prob2=prob2+np.log(float(short))
prob2=np.exp(prob2)*float(counts[counts['word']==pairs2[0].split()[0]]['prob']) 
#9.2022783561072723e-06

prob3=0.0
for item in pairs3:
    short=bigrammodel[bigrammodel['pairs']==item]['con_p']
    if (len(short)>0):
        prob3=prob3+np.log(float(short))
prob3=np.exp(prob3)*float(counts[counts['word']==pairs3[0].split()[0]]['prob']) 
#3.6785901090030055e-09

# Method 3: Bigram maximum likelihood model with add-one smoothing
prob1=0.0
for item in pairs1:
    short=bigrammodel[bigrammodel['pairs']==item]['AddOne_con_p']
    if (len(short)>0):
        prob1=prob1+np.log(float(short))
prob1=np.exp(prob1)*float(counts[counts['word']==pairs1[0].split()[0]]['prob'])
 #7.1458680091693923e-09

prob2=0.0
for item in pairs2:
    short=bigrammodel[bigrammodel['pairs']==item]['AddOne_con_p']
    if (len(short)>0):
        prob2=prob2+np.log(float(short))
prob2=np.exp(prob2)*float(counts[counts['word']==pairs2[0].split()[0]]['prob']) 
#8.639281366256842e-09

prob3=0.0
for item in pairs3:
    short=bigrammodel[bigrammodel['pairs']==item]['AddOne_con_p']
    if (len(short)>0):
        prob3=prob3+np.log(float(short))
prob3=np.exp(prob3)*float(counts[counts['word']==pairs3[0].split()[0]]['prob']) 
#7.3536469205919274e-16

#4 a bigram model with discounting and Katz backoff.
prob1=0.0
for item in pairs1:
    short=bigrammodel[bigrammodel['pairs']==item]['con_p']
    short1=counts[counts['word']==item]['prob']
    if (len(short)>0):
        prob1=prob1+np.log(float(short))
    elif (len(short1)>0):
        pieces=item.split()
        prob1=prob1+np.log(float(counts[counts['word']==pieces[1]]['prob']))
        prob1=prob1+np.log(float(alpha_df[alpha_df['FirstWord']==pieces[1]]['alpha1']))
prob1=np.exp(prob1)*float(counts[counts['word']==pairs1[0].split()[0]]['prob']) 
#3.2724373802806885e-05
#perplexity=1284.8554817842842
prob2=0.0
for item in pairs2:
    short=bigrammodel[bigrammodel['pairs']==item]['con_p']
    short2=counts[counts['word']==item]['prob']
    if (len(short)>0):
        prob2=prob2+np.log(float(short))
    elif (len(short2)>0):
        pieces=item.split()
        prob2=prob2+np.log(float(counts[counts['word']==pieces[1]]['prob']))
        prob2=prob2+np.log(float(alpha_df[alpha_df['FirstWord']==pieces[1]]['alpha1']))
prob2=np.exp(prob2)*float(counts[counts['word']==pairs2[0].split()[0]]['prob']) 
#9.2022783561072723e-06
#perplexity=3095.7203931464346
prob3=0.0
for item in pairs3:
    short=bigrammodel[bigrammodel['pairs']==item]['con_p']
    short3=counts[counts['word']==item]['prob']
    if (len(short)>0):
        prob3=prob3+np.log(float(short))
    elif (len(short3)>0):
        pieces=item.split()
        prob3=prob3+np.log(float(counts[counts['word']==pieces[1]]['prob']))
        prob3=prob3+np.log(float(alpha_df[alpha_df['FirstWord']==pieces[1]]['alpha1']))
prob3=np.exp(prob3)*float(counts[counts['word']==pairs3[0].split()[0]]['prob']) 
#3.6785901090030055e-09
#2**(-np.log(prob3)) perplexity=701817.44363438967


