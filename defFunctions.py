import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

#feature selction
def findCommonFeature(featureLIST):
    featureSET = []
    for item in featureLIST:
        featureSET.append(set(item))

    setItem = featureSET.pop()
    for item in featureSET:
        setItem = setItem.intersection(item)

    return setItem


def findIntersectionFeature(Xfeature, commonFeatures):
    intersect = set(Xfeature) - set(commonFeatures)
    return intersect


def combinedWithFeature(rank, data):
    dictionary = {}
    index = 0
    for feature in data.columns:
        dictionary[feature] = rank[index]
        index = index + 1
    return dictionary


def getFeatureName(dict, asc):
    sortedItem = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]))
    topFeatures = []
    for key, item in sortedItem:
        topFeatures.append(key)
    if(asc != True):
        topFeatures.reverse()
    return topFeatures


def recursiveFeatureElimination(X, Y, featureTaken):
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=featureTaken)
    fit = rfe.fit(X, Y)
    rank = fit.ranking_
    rankedFeature = combinedWithFeature(rank, X)
    topFeatures = getFeatureName(rankedFeature, True)
    return topFeatures[:featureTaken]


def featureExtraction(X, Y, featureTaken):
    model = ExtraTreesClassifier()
    fit = model.fit(X, Y)
    rank = fit.feature_importances_
    rankedFeature = combinedWithFeature(rank, X)
    topFeatures = getFeatureName(rankedFeature, False)
    return topFeatures[:featureTaken]


def univariateFeatureExtraction(X, Y, featureTaken):
    model = SelectKBest(score_func=chi2, k=featureTaken)
    fit = model.fit(X, Y)
    rank = fit.scores_
    rankedFeature = combinedWithFeature(rank, X)
    topFeatures = getFeatureName(rankedFeature, False)
    return topFeatures[:featureTaken]



def mappingNum(column):
    arr = column.unique()
    index = np.argwhere(arr == '?')
    arr = np.delete(arr, index)
    number = 1
    for name in arr:
        column = column.replace({name : number})
        number += 1
    return column

def replaceByMean(column):
    column = pd.to_numeric(column, errors='coerce')
    meanValue = int(round(column.mean()))
    column = column.fillna(meanValue).astype(int)
    return column

    

#Algorithm
def decesionTree(X_train, Y_train):
    dtree = DecisionTreeClassifier()
    return dtree.fit(X_train, Y_train)


def naiveBayes(X_train, Y_train):
    nBayes = MultinomialNB()
    return nBayes.fit(X_train, Y_train)


def adaBoost(X_train, Y_train):
    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    return abc.fit(X_train, Y_train)