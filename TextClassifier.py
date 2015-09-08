# -*- coding: utf-8 -*-
'''
Author: Linjun Li
Data is drawn from '20 Newsgroups', I chose only 2 group for classification.
Training Data vs Testing Data: 5 vs 1 
Class 1: Email requests (Internet news) with the topic of 'sci.crypt'
Class 2: Email requests (Internet news) with the topic of 'sci.electronics'
The following algorithm also works when there are more than two classes.
'''
email linjunli@vt.edu for code.

'''
Test results:
Score with Nearest Neighbors = 0.852130325815
Score with Linear SVM = 0.937343358396
Score with RBF SVM = 0.503759398496
Score with Decision Tree = 0.766917293233
Score with Random Forest = 0.676691729323
Score with AdaBoost = 0.887218045113
Score with Naive Bayes = 0.588972431078

'''
