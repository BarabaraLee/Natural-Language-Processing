# -*- coding: utf-8 -*-
from random import shuffle
from nltk.tokenize import RegexpTokenizer
import os
from nltk.stem.porter import *
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

def rawsample(path,a):
    f = open(path+a, 'r+')
    #w=open(path+'51120n','w+')
    text=f.read()
    try:
        Linepos=text.index("Line")
    except:
        Linepos=0
    start=text[Linepos:].index('\n')
    text=text[start+Linepos+1:]
    #f.write(text)
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    words = ' '.join([w.lower() for w in tokens])
    words = re.findall(' [a-z]+', words)
    words=[w[1:] for w in words]
    #w.write(raw)
    
    stemmer = PorterStemmer()
    try:
        words1 = [stemmer.stem(plural) for plural in words]
    except:
        words1=[]
    return words1
    #return dict(FreqDist(words1))

def union(a, b):
    return set(a+b)

def vpool(path,filenames):
    vocabularypool=[]
    for n in filenames:
        vocabularypool=list(union(vocabularypool,rawsample(path,n)))
    return set(vocabularypool)
        
def totalvpool(totalpath,totalfilenames):
    totalpool=[]
    for i in totalfilenames:
        path=totalpath+i+'/'
        filenames=os.listdir(path)[1:]
        totalpool=list(union(totalpool,list(vpool(path,filenames))))
    return totalpool
        
def featureMatrix(totalpath,totalfilenames):
    totalvocab=totalvpool(totalpath,totalfilenames)
    featurematrix=[]
    for i in totalfilenames:
        path=totalpath+i+'/'
        filenames=os.listdir(path)[1:]
        for a in filenames:
            init=dict([[w,0] for w in totalvocab])
            words1=rawsample(path,a)
            for word in words1:
                init[word]=init.get(word,0)+1
            featurematrix.append(init)
    return featurematrix
    
def fMLabels(totalpath,totalfilenames):
    Labels=[]
    for i in totalfilenames:
        path=totalpath+i+'/'
        filenames=os.listdir(path)[1:]
        for a in filenames:
            Labels.append(i)
    return Labels
            
        
totalpath='/Users/linjunli/Desktop/20_newsgroups 2/'
totalfilenames=os.listdir(totalpath)[1:-1]

holder=featureMatrix(totalpath,totalfilenames)
fMatrix=[t.values() for t in holder]
svd = TruncatedSVD(n_components=100)#n_components can be adjusted
#print(svd.explained_variance_ratio_)
fMsvd=list(svd.fit_transform(fMatrix))
Labels=fMLabels(totalpath,totalfilenames)
c = list(zip(fMsvd, Labels))
shuffle(c)
fMsvd, Labels = zip(*c)

X_train=fMsvd[:1600]
y_train=Labels[:1600]
X_test=fMsvd[1600:]
y_test=Labels[1600:]
#len(Labels)=1999


#Test of "Nearest Neighbors", "Linear SVM", "RBF SVM",... classifiers
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.5),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    print 'Score with '+name+' =',clf.score(X_test, y_test)

'''
Training Data vs Testing Data: 5 vs 1 
Class 1: Email requests with topic of 'sci.crypt'
Class 2: Email requests with topic of 'sci.crypt'

Test results:
Score with Nearest Neighbors = 0.852130325815
Score with Linear SVM = 0.937343358396
Score with RBF SVM = 0.503759398496
Score with Decision Tree = 0.766917293233
Score with Random Forest = 0.676691729323
Score with AdaBoost = 0.887218045113
Score with Naive Bayes = 0.588972431078

'''