import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import XGBoostClassifier as xg
import os
import libavito
import feather

# bagger for xgboost
def bagged_set(X_t,y_c,model, seed, estimators, xt, update_seed=True):

   # create array object to hold predictions
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
        #shuff;e first, aids in increasing variance and forces different results
        #X_t,y_c=shuffle(X,y, random_state=seed+n)

        if update_seed: # update seed if requested, to give a slightly different model
            model.set_params(random_state=seed + n)
        model.fit(X_t,y_c) # fit model0.0917411475506
        preds=model.predict(xt) # predict probabilities
        # update bag's array
        for j in range (0, (xt.shape[0])):
                baggedpred[j]+=preds[j]
        print("done bag %d " % (n))
   # divide with number of bags to create an average estimate
   for j in range (0, len(baggedpred)):
                baggedpred[j]/=float(estimators)
   # return probabilities
   return np.array(baggedpred)




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


def printfilcsve(X, filename):

    np.savetxt(filename,X, fmt='%.5f')


# read the train and test allclean.csv files. skip errors
def readfile(name, index=0):
    dopen=open(name,"r")
    array=[]
    skip_firstrow=False
    if index!=0:
        skip_firstrow=True
    for i,line in enumerate(dopen):
        if i==0 and skip_firstrow:
            continue
        splits=line.replace("\n","").replace(" ","").split(",")
        ar=[]
        for k in splits:
            try:
               ar.append(float(k))
            except:
                ar.append(0.0)
                print(" the string is %s ok?" % ((k)))
        array.append(ar)#[float(k)0.971474 if k!="0" else 0.0 for k in splits ])
        if i%100000==0:
            print(" we are at " , str(i))
    return   np.array(array)


def main():

        config = libavito.get_config()
        cache_loc = config.cache_loc
        nthreads = config.nthreads

        Usecv=True # true will split the training data 66-33 and do cv
        SEED=15
        threads=nthreads # number of workers for parallelism

        ######### Load files ############
        print("Loading input data")
        train = feather.read_dataframe(cache_loc + 'final_featureSet_train.fthr')
        y = train['isDuplicate'].values
        X = train.drop(['itemID_1', 'itemID_2', 'isDuplicate'], 1).values
        del train
        print(X.shape)
        test = feather.read_dataframe(cache_loc + 'final_featureSet_test.fthr')
        ids = test['id'].values
        X_test = test.drop(['itemID_1', 'itemID_2', 'id'], 1).values
        del test
        print(X_test.shape)


        metafolder=cache_loc + "meta_folder/" # folder to use to store for meta predictions
        if not os.path.exists(metafolder):      #if it does not exists, we create it
            os.makedirs(metafolder)
        outset="marios_xgreg_v3" # predic of all files

        #model to use

        idex1=[k for k in range( 0,(X.shape[0] * 2)/ 3)]
        idex2=[k for k in range( (X.shape[0] * 2)/ 3,X.shape[0] )]
        kfolder=[[idex1,idex2]]
        #Create Arrays for meta
        train_stacker=[ 0.0  for k in range (0,len(idex2)) ]
        test_stacker=[0.0  for k in range (0,(X_test.shape[0]))]
        # CHECK EVerything in five..it could be more efficient

        #create target variable
        mean_kapa = 0.0
        #kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=SEED)
        #number_of_folds=0
        #X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time
        i=0 # iterator counter
        if Usecv:
            print ("starting cross validation")
            for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))


                preds=bagged_set(X_train,y_train,model, SEED, 5, X_cv, update_seed=True)


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
            if Usecv:
                print (" Average AUC: %f" % (mean_kapa) )
                print (" printing train datasets ")
                printfilcsve(np.array(train_stacker), metafolder+ outset + "train.csv")


        #preds=bagged_set(X, y,model, SEED, 1, X_test, update_seed=True)

        preds=bagged_set(X, y,model, SEED , 5, X_test, update_seed=True)


        for pr in range (0,len(preds)):
                    test_stacker[pr]=(preds[pr])

        preds=np.array(preds)
        printfilcsve(np.array(test_stacker), metafolder+ outset + "test.csv")


        print("Write results...")
        output_file = "submission_"+ outset +str( (mean_kapa ))+".csv"
        print("Writing submission to %s" % output_file)
        f = open(config.output_loc + output_file, "w")
        f.write("id,probability\n")# the header
        for g in range(0, len(preds))  :
            pr=preds[g]
            f.write("%d,%f\n" % (((ids[g]),pr ) ) )
        f.close()
        print("Done.")






if __name__=="__main__":
  main()
