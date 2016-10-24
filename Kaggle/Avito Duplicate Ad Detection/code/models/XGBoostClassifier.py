# -*- coding: utf-8 -*-
"""
Created on oct 20 23:15:24 2015

@author: marios

Script that makes Xgboost scikit-like.

The initial version of the script came from Guido Tapia (or such is his kaggle name!). I have modified it quite a bit though.

the github from where this was retrieved was : https://github.com/gatapia/py_ml_utils

He has done excellent job in making many commonly used algorithms scikit-like 

"""

# licence: FreeBSD

"""
Copyright (c) 2015, Marios Michailidis
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import sys
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix


def softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score-np.max(score))
    score /= np.sum(score, axis=1)[:,np.newaxis]
    return score

## soft version of kappa score using the class probability
## inspired by @George Mohler in the Higgs competition
## https://www.kaggle.com/c/higgs-boson/forums/t/10286/customize-loss-function-in-xgboost/53459#post53459
## NOTE: As also discussed in the above link, it is hard to tune the hessian to get it to work.
def softkappaobj(preds, dtrain):
    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]

    ## compute O (enumerator)
    O = 0.0
    for j in range(N):
        wj = (labels - (j+1.))**2
        O += np.sum(wj * preds[:,j])
    
    ## compute E (denominator)
    hist_label = np.bincount(labels)[1:]
    hist_pred = np.sum(preds, axis=0)
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]

    ## compute gradient and hessian
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        ## first-order derivative: dO / dy_mn
        dO = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            dO += ((labels - (j+1.))**2) * preds[:,n] * (indicator - preds[:,j])
        ## first-order derivative: dE / dy_mn
        dE = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                dE += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (indicator - preds[:,k])
        ## the grad
        grad[:,n] = -M * (dO * E - O * dE) / (E**2)
        
        ## second-order derivative: d^2O / d (y_mn)^2
        d2O = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            d2O += ((labels - (j+1.))**2) * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,j])
       
        ## second-order derivative: d^2E / d (y_mn)^2
        d2E = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                d2E += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,k])
        ## the hess
        hess[:,n] = -M * ((d2O * E - O * d2E)*(E**2) - (dO * E - O * dE) * 2. * E * dE) / (E**4)

    grad *= -1.
    hess *= -1.
    # this pure hess doesn't work in my case, but the following works ok
    # use a const
    #hess = 0.000125 * np.ones(grad.shape, dtype=float)
    # or use the following...
    scale = 0.000125 / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess) # It works!! no idea...
    grad.shape = (M*N)
    hess.shape = (M*N)
    return grad, hess

# evalerror is your customized evaluation function to 
# 1) decode the class probability 
# 2) compute quadratic weighted kappa
def evalerror(preds, dtrain):
    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
    labels = dtrain.get_label() + 1
    ## class probability
    preds = softmax(preds)
    ## decoding (naive argmax decoding)
    pred_labels = np.argmax(preds, axis=1) + 1
    ## compute quadratic weighted kappa (using implementation from @Ben Hamner
    ## https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
    kappa = quadratic_weighted_kappa(labels, pred_labels)
    return 'kappa', kappa


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, silent=True,
      use_buffer=True, num_round=10,num_parallel_tree=1, ntree_limit=0,
      nthread=None, booster='gbtree', 
      eta=0.3, gamma=0.01, 
      max_depth=6, min_child_weight=1, subsample=1, 
      colsample_bytree=1,      
      l=0, alpha=0, lambda_bias=0, objective='reg:linear',
      eval_metric='logloss', seed=0, num_class=None,
      max_delta_step=0,classes_=None ,
      colsample_bylevel=1.0 , sketch_eps=0.1 , sketch_ratio=2.0 ,
      opt_dense_col=1, size_leaf_vector=0.0, min_split_loss=0.0,
      cache_opt=1, default_direction =0 , k_folds=0 ,early_stopping_rounds=200 
      ):    
    assert booster in ['gbtree', 'gblinear']
    assert objective in ['reg:linear', 'reg:logistic', 
      'binary:logistic', 'binary:logitraw', 'multi:softmax',
      'multi:softprob', 'rank:pairwise','count:poisson']
    assert eval_metric in [ 'rmse', 'mlogloss', 'logloss', 'error', 
      'merror',  'auc', 'ndcg', 'map', 'ndcg@n', 'map@n', 'kappa']
    if eval_metric=='kappa':
        booster='gblinear'
    self.silent = silent
    self.use_buffer = use_buffer
    self.num_round = num_round
    self.ntree_limit = ntree_limit
    self.nthread = nthread 
    self.booster = booster
    # Parameter for Tree Booster
    self.eta=eta
    self.gamma=gamma
    self.max_depth=max_depth
    self.min_child_weight=min_child_weight
    self.subsample=subsample
    self.colsample_bytree=colsample_bytree
    self.colsample_bylevel=colsample_bylevel
    self.max_delta_step=max_delta_step
    self.num_parallel_tree=num_parallel_tree
    self.min_split_loss=min_split_loss
    self.size_leaf_vector=size_leaf_vector
    self.default_direction=default_direction
    self.opt_dense_col=opt_dense_col
    self.sketch_eps=sketch_eps
    self.sketch_ratio=sketch_ratio
    self.k_folds=k_folds
    self.k_models=[]
    self.early_stopping_rounds=early_stopping_rounds
    
    # Parameter for Linear Booster
    self.l=l
    self.alpha=alpha
    self.lambda_bias=lambda_bias
    # Misc
    self.objective=objective
    self.eval_metric=eval_metric
    self.seed=seed
    self.num_class = num_class
    self.n_classes_ =num_class
    self.classes_=classes_
    

  def set_params(self,random_state=1):
      self.seed=random_state
      
  def build_matrix(self, X, opt_y=None, weighting=None):
    if opt_y==None: 
        if weighting==None:
            return xgb.DMatrix(csr_matrix(X), missing =-999.0)
        else :
            #scale weight
            sumtotal=float(X.shape[0])
            sumweights=np.sum(weighting)            
            for s in range(0,len(weighting)):
                weighting[s]*=sumtotal/sumweights
            return xgb.DMatrix(csr_matrix(X), missing =-999.0, weight=weighting)            
    else:
        if weighting==None:           
            return xgb.DMatrix(csr_matrix(X), label=np.array(opt_y), missing =-999.0)
        else :
            sumtotal=float(X.shape[0])
            sumweights=np.sum(weighting)            
            for s in range(0,len(weighting)):
                weighting[s]*=sumtotal/sumweights             
            return xgb.DMatrix(csr_matrix(X), label=np.array(opt_y), missing =-999.0, weight=weighting)         


  
  def fit(self, X, y,sample_weight=None):    
    
    self.k_models=[]
    
    X1 = self.build_matrix(X, y,weighting= sample_weight)#sample_weight)
    param = {}
    param['booster']=self.booster
    param['objective'] = self.objective
    param['bst:eta'] = self.eta
    param['seed']=  self.seed  
    param['bst:max_depth'] = self.max_depth
    if self.eval_metric!='kappa':
        param['eval_metric'] = self.eval_metric
    param['bst:min_child_weight']= self.min_child_weight
    param['silent'] =  1  if self.silent==True else 0
    param['nthread'] = self.nthread
    param['bst:subsample'] = self.subsample 
    param['subsample'] = self.subsample
    param['min_child_weight'] = self.min_child_weight
    param['max_depth'] = self.max_depth
    param['eta'] = self.eta
    param['gamma'] = self.gamma
    param['colsample_bytree']= self.colsample_bytree    
    param['num_parallel_tree']= self.num_parallel_tree   
    #param['colsample_bylevel']= self.colsample_bylevel             
    #param['min_split_loss']=self.min_split_loss
    #param['default_direction']=self.default_direction    
    #param['opt_dense_col']=self.opt_dense_col        
    #param['sketch_eps']=self.sketch_eps    
    #param['sketch_ratio']=self.sketch_ratio            
    #param['size_leaf_vector']=self.size_leaf_vector 

    if self.num_class is not None:
      param['num_class']= self.num_class
    if self.k_folds <2:
         if self.eval_metric!='kappa':
            self.bst = xgb.train(param.items(), X1, self.num_round)
         else :
            self.bst = xgb.train(param.items(), X1, self.num_round, obj=softkappaobj, feval=evalerror)
    else :
        number_of_folds=self.k_folds
        kfolder2=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=self.seed)
        ## we split 64-16 5 times to make certain all the data has been use in modelling at least once 
        for train_indexnew, test_indexnew in kfolder2:
            if sample_weight==None:
                dtrain = xgb.DMatrix(X[train_indexnew], label=y[train_indexnew])
                dtvalid = xgb.DMatrix(X[test_indexnew], label=y[test_indexnew])
            else :
                dtrain = xgb.DMatrix(X[train_indexnew], label=y[train_indexnew], weight=sample_weight[train_indexnew])
                dtvalid = xgb.DMatrix(X[test_indexnew], label=y[test_indexnew], weight=sample_weight[test_indexnew])  
            
            watchlist = [(dtrain, 'train'), (dtvalid, 'valid')]
            if self.eval_metric!='kappa':
                gbdt = xgb.train(param.items(), dtrain, self.num_round, watchlist, verbose_eval=False, early_stopping_rounds=self.early_stopping_rounds)#, verbose_eval=250) #, early_stopping_rounds=250, verbose_eval=250) 
            else :
                gbdt = xgb.train(param.items(), dtrain, self.num_round, watchlist, verbose_eval=False, obj=softkappaobj, feval=evalerror, early_stopping_rounds=self.early_stopping_rounds)#, verbose_eval=250) #, early_stopping_rounds=250, verbose_eval=250) 
                
           #predsnew = gbdt.predict(dtest, ntree_limit=gbdt.best_iteration)  
            self.k_models.append(gbdt)

    return self

  def predict(self, X): 
    if  self.k_models!=None and len(self.k_models)<2:
        X1 = self.build_matrix(X)
        return self.bst.predict(X1)
    else :
        dtest = xgb.DMatrix(X)
        preds= [0.0 for k in X.shape[0]]
        for gbdt in self.k_models:
            predsnew = gbdt.predict(dtest, ntree_limit=(gbdt.best_iteration+1)*self.num_parallel_tree)  
            for g in range (0, predsnew.shape[0]):
                preds[g]+=predsnew[g]
        for g in range (0, len(preds)):
            preds[g]/=float(len(self.k_models))       
  
  def predict_proba(self, X): 
    try:
      rows=(X.shape[0])
    except:
      rows=len(X)
    X1 = self.build_matrix(X)
    if  self.k_models!=None and len(self.k_models)<2:
        predictions = self.bst.predict(X1)
    else :
        dtest = xgb.DMatrix(X)
        predictions= None
        for gbdt in self.k_models:
            predsnew = gbdt.predict(dtest, ntree_limit=(gbdt.best_iteration+1)*self.num_parallel_tree)  
            if predictions==None:
                predictions=predsnew
            else:
                for g in range (0, predsnew.shape[0]):
                    predictions[g]+=predsnew[g]
        for g in range (0, len(predictions)):
            predictions[g]/=float(len(self.k_models))               
        predictions=np.array(predictions)
    if self.objective == 'multi:softprob': return predictions.reshape( rows, self.num_class)
    return np.vstack([1 - predictions, predictions]).T
    
def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat
    
def quadratic_weighted_kappa(rater_a, rater_b, min_rating=1, max_rating=8):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator
