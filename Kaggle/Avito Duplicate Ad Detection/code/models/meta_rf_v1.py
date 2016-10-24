# -*- coding: utf-8 -*-
"""
@author: marios

Script that does meta modelling level 1 as in taking the held-out predictions from the previous models and using as features in a new model.

This one uses xgboost to do this.

"""

import numpy as np
import gc
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import XGBoostClassifier as xg
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import libavito
import feather


#load a single file

def loadcolumn(filename,col=4, skip=1, floats=True):
    pred=[]
    op=open(filename,'r')
    if skip==1:
        op.readline() #header
    for line in op:
        line=line.replace('\n','')
        sps=line.split(',')
        #load always the last columns
        if floats:
            pred.append(float(sps[col]))
        else :
            pred.append(str(sps[col]))
    op.close()
    return pred


#functions to manipulate pickles

def load_datas(filename):

    return joblib.load(filename)

def printfile(X, filename):

    joblib.dump((X), filename)

def printfilcsve(X, filename):

    np.savetxt(filename,X)



def bagged_set(X,y,model, seed, estimators, xt, update_seed=True):

   # create array object to hold predictions
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
        #shuff;e first, aids in increasing variance and forces different results
        X_t,y_c=shuffle(X,y, random_state=seed+n)

        if update_seed: # update seed if requested, to give a slightly different model
            model.set_params(random_state=seed + n)
        model.fit(X_t,y_c) # fit model0.0917411475506
        preds=model.predict_proba(xt)[:,1] # predict probabilities
        # update bag's array
        for j in range (0, (xt.shape[0])):
                baggedpred[j]+=preds[j]
   # divide with number of bags to create an average estimate
   for j in range (0, len(baggedpred)):
                baggedpred[j]/=float(estimators)
   # return probabilities
   return np.array(baggedpred)



def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def main():
        config = libavito.get_config()
        nthreads = config.nthreads
        cache_loc = config.cache_loc
        output_loc = config.output_loc

        load_data=True
        SEED=15
        Usecv=True
        meta_folder=cache_loc + "meta_folder/" # this is how you name the foler that keeps the held-out and test predictions to be used later for meta modelling
        # least of meta models. All the train held out predictions end with a 'train.csv' notation while all test set predictions with a 'test.csv'
        meta=['marios_xg_v1','marios_nn_v1',
              'marios_nnnew_v2','marios_sgd_v2','marios_logit_v2','marios_ridge_v2','marios_xgson_v2','marios_xgrank_v2',
              'marios_xgson_v3','marios_nnnew_v3','marios_xgrank_v3','marios_xgreg_v3',
              'marios_nnnew_v4','marios_xgson_v4',
              'marios_xgsonv2_v5']#,'marios_xgsonv2_v5'

        bags=5 # helps to avoid overfitting. Istead of 1, we ran 10 models with differnt seed and different shuffling
        ######### Load files (...or not!) ############

        #y = np.loadtxt(meta_folder+"meta_pairs_and_labels.csv", delimiter=',',usecols=[2], skiprows=1)
        #ids=np.loadtxt("Random_submission.csv", delimiter=',',usecols=[0], skiprows=1)
        print("Loading input data")
        train = feather.read_dataframe(cache_loc + 'final_featureSet_train.fthr')
        y = train['isDuplicate'].values
        del train
        test = feather.read_dataframe(cache_loc + 'final_featureSet_test.fthr')
        ids = test['id'].values
        del test

        # the trainstacked features is a dataset provided from Henk and Mathes that contains:
        #a couple of FTRL models, non alphanumeric , sentiment scores and some additional models
        if load_data:
            Xmetatrain=None
            Xmetatest=None
            for modelname in meta :
                    mini_xtrain=np.loadtxt(meta_folder + modelname + 'train.csv') # we load the held out prediction of the int'train.csv' model
                    mini_xtest=np.loadtxt(meta_folder + modelname + 'test.csv')   # we load the test set prediction of the int'test.csv' model
                    mean_train=np.mean(mini_xtrain) # we calclaute the mean of the train set held out predictions for reconciliation purposes
                    mean_test=np.mean(mini_xtest)    # we calclaute the mean of the test set  predictions
                    # we print the AUC and the means and we still hope that everything makes sense. Eg. the mean of the train set preds is 1232314.34 and the test is 0.7, there is something wrong...
                    print("model %s auc %f mean train/test %f/%f " % (modelname,roc_auc_score(np.array(y),mini_xtrain) ,mean_train,mean_test))
                    if Xmetatrain==None:
                        Xmetatrain=mini_xtrain
                        Xmetatest=mini_xtest
                    else :
                        Xmetatrain=np.column_stack((Xmetatrain,mini_xtrain))
                        Xmetatest=np.column_stack((Xmetatest,mini_xtest))
            # we combine with the stacked features
            X=Xmetatrain
            X_test=Xmetatest
            # we print the pickles
            printfile(X,meta_folder+"xmetahome.pkl")
            printfile(X_test,meta_folder+"xtmetahome.pkl")

            X=load_datas(meta_folder+"xmetahome.pkl")
            print("rows %d columns %d " % (X.shape[0],X.shape[1] ))
            #X_test=load_datas("onegramtest.pkl")
            #print("rows %d columns %d " % (X_test.shape[0],X_test.shape[1] ))
        else :

            X=load_datas(meta_folder+"xmetahome.pkl")
            print("rows %d columns %d " % (X.shape[0],X.shape[1] ))
            X_test=load_datas(meta_folder+"xtmetahome.pkl")
            print("rows %d columns %d " % (X_test.shape[0],X_test.shape[1] ))


        outset="marios_rf_meta_v1" # Name of the model (quite catchy admitedly)


        print("len of target=%d" % (len(y))) # print the length of the target variable because we can

        #model we are going to use
        #ExtraTreesClassifier

        model=RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=9,  min_samples_leaf=2, max_features=8, n_jobs=nthreads, random_state=1, verbose=1)

        #model=LogisticRegression(C=0.01)
        train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] # the object to hold teh held-out preds

        idex1=[k for k in range( 0,(X.shape[0] * 2)/ 3)] # indices for trai
        idex2=[k for k in range( (X.shape[0] * 2)/ 3,X.shape[0] )]  #indices for test
        kfolder=[[idex1,idex2]] # create an object to put indices in

        #arrays to save predictions for validation and test for meta modelling (stacking)
        train_stacker=[ 0.0  for k in range (0,len(idex2)) ]
        test_stacker=[0.0  for k in range (0,(X_test.shape[0]))]

        #create target variable
        mean_kapa = 0.0
        #X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time
        i=0 # iterator counter
        if Usecv:
            print ("starting cross validation" )
            for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))

                #use xgboost bagger
                preds=bagged_set(X_train,y_train,model, SEED, bags, X_cv, update_seed=True)

                # compute Loglikelihood metric for this CV fold
                #scalepreds(preds)
                kapa = roc_auc_score(y_cv,preds)
                print "size train: %d size cv: %d AUC (fold %d/%d): %f" % ((X_train.shape[0]), (X_cv.shape[0]), i + 1, 1, kapa)

                mean_kapa += kapa
                #save the results
                no=0
                for real_index in test_index:
                         train_stacker[no]=(preds[no])
                         no+=1
                i+=1
            if (Usecv):
                #print the array of validation predictions for stacking later on inside the 'meta_folder'
                print (" Average AUC: %f" % (mean_kapa) )
                print (" printing train datasets ")
                printfilcsve(np.array(train_stacker), meta_folder+ outset + "train.csv")

        preds=bagged_set(X, y,model, SEED ,bags, X_test, update_seed=True)


        for pr in range (0,len(preds)):
                    test_stacker[pr]=(preds[pr])
        #print prediction as numpy array for stacking later on
        preds=np.array(preds)
        printfilcsve(np.array(test_stacker), meta_folder+ outset + "test.csv")

        #create submission file
        print("Write results...")
        output_file = "submission_"+ outset +str( (mean_kapa ))+".csv"
        print("Writing submission to %s" % output_file)
        f = open(output_loc + output_file, "w")
        f.write("id,probability\n")# the header
        for g in range(0, len(preds))  :
            pr=preds[g]
            f.write("%d,%f\n" % (((ids[g]),pr ) ) )
        f.close()
        print("Done.")





if __name__=="__main__":
  main()
