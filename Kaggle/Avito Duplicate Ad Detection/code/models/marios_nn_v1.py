import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import *
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras import optimizers
from sklearn import cross_validation
import keras
import libavito
import feather


epc=150
batch=1024

def build_model(input_dim, output_dim):
    models = Sequential()

    models.add(Dense(2000, input_dim=input_dim, init='uniform', W_regularizer=l2(0.00001)))
    models.add(PReLU())
    models.add(BatchNormalization())
    models.add(Dropout(0.6))
    models.add(Dense(output_dim, init='uniform'))
    models.add(Activation('softmax'))

    opt = optimizers.Adagrad(lr=0.01)
    models.compile(loss='binary_crossentropy', optimizer=opt)
    return models


def bagged_set(X,y, seed, estimators, xt, nval=0.0, verbos=0):

   baggedpred=[ 0.0 for d in range(0, xt.shape[0])]

   for i in range (0, estimators):

        X_t,y_c=shuffle(X,y, random_state=seed+i)
        np.random.seed(seed+i)
        model = build_model(xt.shape[1], 2)
        if nval>0.0:
            x_train_oof, x_valid_oof, y_train_oof_nn, y_valid_oof_nn = cross_validation.train_test_split(
            X_t, y_c, test_size=nval, random_state=i*seed)
            model.fit(x_train_oof,np_utils.to_categorical( y_train_oof_nn),
                      nb_epoch=epc, batch_size=batch,
                      validation_data=(x_valid_oof,np_utils.to_categorical(y_valid_oof_nn)),
                    verbose=verbos, callbacks=[MonitorAUC(x_valid_oof,y_valid_oof_nn)])
        else :
            y_train_oof_nn = np_utils.to_categorical(y_c)
            model.fit(X_t, y_train_oof_nn, nb_epoch=epc, batch_size=batch, verbose=verbos)

        preds =model.predict_proba(xt) [:,1]
        for j in range (0, xt.shape[0]):
                 baggedpred[j]+=preds[j]

   for j in range (0, len(baggedpred)):
                baggedpred[j]/=float(estimators)

   return np.array(baggedpred)


class MonitorAUC(keras.callbacks.Callback):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs={}):
        yhat = self.model.predict_proba(self.x, verbose=0).T[1]
        print 'AUC', roc_auc_score(self.y, yhat)



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

        Use_scale=True
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
        #remove begatives
        X[ X < 0] = 0
        X_test[ X_test < 0] = 0
        #transform the data with log1p
        X=np.log1p(X)
        X_test=np.log1p(X_test)

        #create meta folder to drop predictions for train and test
        if not os.path.exists(metafolder):      #if it does not exists, we create it
            os.makedirs(metafolder)


        outset="marios_nn_v1" #prefix
        #model to use

        idex1=[k for k in range( 0,(X.shape[0] * 2)/ 3)]
        idex2=[k for k in range( (X.shape[0] * 2)/ 3,X.shape[0] )]
        kfolder=[[idex1,idex2]]
        #Create Arrays for meta
        train_stacker=[ 0.0  for k in range (0,(idex2.shape[0])) ]
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

                if Use_scale:
                    stda=StandardScaler()
                    X_train=stda.fit_transform(X_train)
                    X_cv=stda.transform(X_cv)

                preds=bagged_set(X_train,y_train, SEED, 10, X_cv, nval=0.0, verbos=0)


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

        if Use_scale:
            stda=StandardScaler()
            X=stda.fit_transform(X)
            X_test=stda.transform(X_test)

        #preds=bagged_set(X, y,model, SEED, 1, X_test, update_seed=True)

        preds=bagged_set(X, y, SEED, 10, X_test, nval=0.0, verbos=0)


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
