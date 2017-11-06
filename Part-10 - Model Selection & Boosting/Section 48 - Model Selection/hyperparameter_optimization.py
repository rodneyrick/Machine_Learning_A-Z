#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:19:57 2017

@author: rick
"""

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import matplotlib.pyplot as plt    

def custom_avg_roc_auc_score(truth, pred):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average="macro")
    

def estimator_metrics(estimator, X, y):
    pred = estimator.predict(X)
    prec, rec, f1, supp = precision_recall_fscore_support(y, pred, average=None)
    
    return {'estimator' : estimator.best_estimator_,
            'parameters': estimator.best_params_,
            'best_score': estimator.best_score_,
            'precision' : prec,
            'recall'    : rec,
            'fscore'    : f1,
            'support'   : supp}
    

def search_parameters_by_accuracy(estimator, param_distributions, X, y, cv=10, n_iter=10):
    RANDOM_STATE = 41
    rand_search = RandomizedSearchCV(estimator,
                                    param_distributions,
                                    n_iter=n_iter,
                                    scoring='accuracy',
                                    fit_params=None,
                                    n_jobs=-1,
                                    iid=True,
                                    refit=True,
                                    cv=cv,
                                    verbose=0,
                                    pre_dispatch='2*n_jobs',
                                    random_state=RANDOM_STATE,
                                    error_score='raise')
                                    #return_train_score=True)
    
    rand_search.fit(X, y)
    return estimator_metrics(rand_search, X, y)


def search_parameters_by_roc_auc(estimator, param_distributions, X, y, cv=10, n_iter=10):
    avg_roc_auc_scorer = make_scorer(custom_avg_roc_auc_score)

    RANDOM_STATE = 41
    rand_search = RandomizedSearchCV(estimator,
                                    param_distributions,
                                    n_iter=n_iter,
                                    scoring=avg_roc_auc_scorer,
                                    fit_params=None,
                                    n_jobs=-1,
                                    iid=True,
                                    refit='avg_roc_auc_scorer',
                                    cv=cv,
                                    verbose=0,
                                    pre_dispatch='2*n_jobs',
                                    random_state=RANDOM_STATE,
                                    error_score='raise')
    rand_search.fit(X, y)
    return estimator_metrics(rand_search, X, y)


def test_score(X,y):
    estimator = KNeighborsClassifier(n_jobs=-1)
    param_distributions = {'n_neighbors': list(range(1, 31)),
                            'weights'    : ['uniform', 'distance']}
     
#    scoring = {'precision_recall_fscore_support': \
#                   make_scorer(custom_precision_recall_fscore_support), 
#               'avg_roc_auc_scorer': make_scorer(custom_avg_roc_auc_score)}
    
    #scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    scoring = make_scorer(custom_avg_roc_auc_score)

    RANDOM_STATE = 41
    rand_search = RandomizedSearchCV(estimator,
                                    param_distributions,
                                    n_iter=10,
                                    scoring=scoring,
                                    n_jobs=1,
                                    iid=True,
                                    refit=True,
                                    cv=2,
                                    verbose=0,
                                    pre_dispatch='2*n_jobs',
                                    random_state=RANDOM_STATE,
                                    error_score='raise')
    rand_search.fit(X, y)
    return estimator_metrics(rand_search, X, y)


def get_scores_to_chart(scores):
    names = []
    acc = []
    prec = []
    rec = []
    f1 = []
    for i in scores:
        names.append(i['name'])
        acc.append(i['accuracy'])
        prec.append(np.mean(i['precision']))
        rec.append(np.mean(i['recall']))
        f1.append(np.mean(i['fscore']))
    
    if len(names) < 2:
        acc = [acc]
    
    values = acc + prec + rec + f1
    min_value = np.min(values)
    max_value = np.min(values)
    
    return {
        'names'     : names,
        'accuracy'  : acc,
        'precision' : prec,
        'recall'    : rec,
        'fscore'    : f1,
        'min_value' : min_value,
        'max_value' : max_value
    }


def chart_estimator_score(scores):
    bar_names = scores['names']
    ind = np.arange(len(bar_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='k')
    
    ax.barh(ind          , scores['accuracy'] , width , color='red'   , label='Accuracy')
    ax.barh(ind + width  , scores['precision'], width , color='navy'  , label='Precision')
    ax.barh(ind + 2*width, scores['recall']   , width , color='orange', label='Recall')
    #ax.barh(ind + width, scores['precision'], width , color='black', label='Precision')
    
    ax.set_yticks(ind)
    ax.set_yticklabels(bar_names, fontdict={'fontsize': 20})
    
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    #ax.set_ylim([3*width-1, len(df)])
    #ax.set_xlim([0,1])
    
    min_lim = scores['min_value'] - 0.05
    max_lim = scores['max_value'] + 0.05
    ax.set_xlim([min_lim,max_lim])
    
    ax.legend()
    plt.show()


def compare_models(estimators, X, y, n_iter=10, cv=2):
    scores = []
    for (name, estimator, params) in estimators:
        print(name)
        #s = search_parameters_by_accuracy(estimator, params, X, y, n_iter=30)        
        s = search_parameters_by_roc_auc(estimator, params, X, y, n_iter=n_iter, cv=cv)
        scores.append({
                'name'      : name, 
                'estimator' : s['estimator'], 
                'parameters': s['parameters'],
                'accuracy'  : s['best_score'],
                'precision' : s['precision'],
                'recall'    : s['recall'], 
                'fscore'    : s['fscore'],
                'support'   : s['support']
        })
    return scores



from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# data
# read in the iris data
iris = load_iris()
# create X (features) and y (response)
X = iris.data
y = iris.target

RANDOM_STATE = 41
estimators = [
    ('k-NN', 
     KNeighborsClassifier(n_jobs=-1),
     {'n_neighbors' : list(range(1, 31)),
     'weights'      : ['uniform', 'distance']}),
    
    ('AB+DT', 
     AdaBoostClassifier(base_estimator = \
                            DecisionTreeClassifier(random_state = RANDOM_STATE),
                        algorithm='SAMME.R',
                        random_state=RANDOM_STATE),
     {'base_estimator__criterion' : ["gini", "entropy"],
     'base_estimator__splitter'   : ["best", "random"],
     'n_estimators'               : [10, 20, 30, 40, 50],
     'learning_rate'              : [10, 20, 30, 40, 50]}),
    
    ('Random Forest', 
     RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, verbose=0),
     {'n_estimators'     : list(range(10, 31)),
     "max_depth"         : [3, None],
     "max_features"      : ['sqrt', 'log2', None],
     "min_samples_split" : [2, 3, 10],
     "min_samples_leaf"  : [1, 3, 10],
     "bootstrap"         : [True, False],
     "criterion"         : ["gini", "entropy"]}),

    ('SVM', 
     SVC(random_state=RANDOM_STATE, verbose=0),
     {'C'                      : [0.5, 1.0,],
     'kernel'                  : ['linear', 'poly', 'rbf'],
     'degree'                  : [3],
     'gamma'                   : ['auto'],
     'coef0'                   : [0.0],
     'shrinking'               : [True],
     'probability'             : [False],
     'tol'                     : [0.001],
     'cache_size'              : [200],
     'class_weight'            : [None],
     'max_iter'                : [10,20],
     'decision_function_shape' : ['ovo','ovr']}),
    
    ('Voting (SVM+RF+NB)', 
      VotingClassifier(estimators=[('svm', SVC(random_state=RANDOM_STATE, 
                                               verbose=0,
                                               probability=True)), 
                                   ('rf' , RandomForestClassifier(n_jobs=-1, 
                                                                  random_state=RANDOM_STATE, 
                                                                  verbose=0)), 
                                   ('gnb', GaussianNB())]),
    {'weights': [[1, 1, 5],
                 [1, 2, 5],
                 [2, 2, 4],
                 [3, 2, 4]],
     'voting'          : ['hard', 'soft'],
     'svm__kernel'     : ['linear', 'poly', 'rbf'],
     'svm__C'          : [1.0, 0.5],
     'svm__degree'     : [3],
     'svm__max_iter'   : [10,20, 30],
     'rf__n_estimators': list(range(10, 31)),
     'rf__criterion'   : ["gini", "entropy"],
     'rf__max_features': ['sqrt', 'log2', None]
    }),
    
    ('MLP', 
     MLPClassifier(random_state=RANDOM_STATE,
                   verbose=0),
    {'hidden_layer_sizes' : [(100,32,16), (64,16,8)],
     'activation'          : ['identity', 'logistic', 'tanh', 'relu'],
     'solver'              : ['lbfgs', 'sgd', 'adam'],
     'max_iter'            : list(range(10,200,20)),
     'warm_start'          : [False, True]
    }),
    
    ('xgboost', 
     xgb.XGBClassifier(nthread=4,
                       objective='multi:softprob',
                       seed=RANDOM_STATE),
     {'n_estimators'    : list(range(10,30,5)), 
      'subsample'       : [0.3, 0.5, 0.8], 
      'colsample_bytree': [0.5, 0.8]}),
]



#from sklearn.decomposition import PCA
#pca = PCA(random_state = RANDOM_STATE)
#pca.fit(X)
#number_of_components = np.sum(pca.explained_variance_ratio_ > 0.05)
#pca = PCA(n_components=number_of_components)
#pca.fit(X)
#X = pca.transform(X)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(solver='eigen', priors=[0,1,2], shrinkage='auto')
X = lda.fit_transform(X, y)

# http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
#from sklearn.preprocessing import RobustScaler
#X =  RobustScaler().fit_transform(X)

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

scores = compare_models(estimators, X, y, n_iter=10, cv=2)
points = get_scores_to_chart(scores)
chart_estimator_score(points)

























































