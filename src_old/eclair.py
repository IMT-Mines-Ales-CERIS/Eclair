# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:34:31 2023

@author: imoussaten
"""
# this code is part of uncertainty quantification project
# this contains the code for the eclair method
####################### Evidential CLAssification with Imprecise Relabelling
#######################  A. Imoussaten & L. Jacquin

#%% imports
import tensorflow as tf
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#print("TensorFlow version:", tf.__version__)
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
#from sklearn.neighbors import KNeighborsClassifier
#import math
#from collections import Counter
from scipy.stats import entropy
import random
import itertools
base = 2  # work in units of bits

#file from the uncertainty quantification project
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from util import Util
from setValuedClassification import SVC
from nbc import nbc

#%% class eclair
class Eclair:
    def trainModelkFolds_nbc(train_xdata, train_ydata, cat_feat, num_feat, NB_SPLIT, random_state):
         #split train data to NB_SPLIT blocks
         kf = KFold(n_splits=NB_SPLIT)
         kf.get_n_splits(train_ydata)
         train_k=[]
         test_k=[]

         #KFold retourne un objet dont la fonction split retourne des tableaux d'indice pour le jeu d'entrainement et le jeu de test. Il en retourne autant qu'il y a de NB_SPLIT.

         for i, (train_index, test_index) in enumerate( kf.split(train_ydata) ):
             train_k.append( train_index  )
             test_k.append( test_index )
         #test_k[0] starts to 0, test_k[1] starts to len(train_ydata)/NB_SPLIT, ...
         
         cpc_proba_test=[]
         cpc_proba_cal=[]
         y_cal=[]
         for k in range( NB_SPLIT ):
             X_train, X_cal, y_train, y_cal_tmp = train_test_split(train_xdata[train_k[k], :], train_ydata[train_k[k]], test_size=0.2, random_state=random_state)
             y_cal.append( y_cal_tmp )
             mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(X_train,y_train, cat_feat, num_feat, 'gaussian')
             #print(sigma2)
             #print([nc, classes, col_n])
             u=np.identity(len(classes))
             preds_bo_cal, preds_eu_cal, preds_proba_cal, preds_proba_sm_cal=nbc.predict(X_cal, cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, u)
             preds_bo, preds_eu, preds_proba, preds_proba_sm=nbc.predict(train_xdata[test_k[k], :], cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, u)
             #print(preds_proba)
             cpc_proba_cal.append( preds_proba_sm_cal )
             cpc_proba_test.append( preds_proba_sm )

         return y_cal, cpc_proba_cal, cpc_proba_test
     
    def trainModelkFolds_dnn(train_xdata, train_ydata, ep, nbC, NB_SPLIT):
         #split train data to NB_SPLIT nlocks
         kf = KFold(n_splits=NB_SPLIT)
         kf.get_n_splits(train_ydata)
         train_k=[]
         test_k=[]
         for i, (train_index, test_index) in enumerate( kf.split(train_ydata) ):
             train_k.append( train_index  )
             test_k.append( test_index )
         #test_k[0] starts to 0, test_k[1] starts to len(train_ydata)/NB_SPLIT, ...
         
         cpc_proba_test=[]
         cpc_proba_cal=[]
         y_cal=[]
         for k in range( NB_SPLIT ):
             RANDOM_STATE = 2018
             TEST_SIZE = 0.2
             X_train, X_cal, y_train, y_cal_tmp = train_test_split(train_xdata[train_k[k], :], train_ydata[train_k[k]], test_size=TEST_SIZE, random_state=RANDOM_STATE)
             #m_train=len(y_train) #0.8*len(train_ydata)
             #k_cal=len(y_cal_tmp) #0.2*len(train_ydata)
             y_cal.append( y_cal_tmp )
             ###
             BATCH_SIZE = 128
             IMG_D1=28
             IMG_D2=28
             #Train the model
             ##Build the model
             ### We will use a Sequential model.
             # The Sequential model is a linear stack of layers. It can be first initialized and then we add layers using add method or we can add all layers at init stage. The layers added are as follows:
             # Conv2D is a 2D Convolutional layer (i.e. spatial convolution over images). The parameters used are:
             # filters - the number of filters (Kernels) used with this layer; here filters = 32;
             # kernel_size - the dimmension of the Kernel: (3 x 3);
             # activation - is the activation function used, in this case relu;
             # kernel_initializer - the function used for initializing the kernel;
             # input_shape - is the shape of the image presented to the CNN: in our case is 28 x 28 The input and output of the Conv2D is a 4D tensor.
             # MaxPooling2D is a Max pooling operation for spatial data. Parameters used here are:
             # pool_size, in this case (2,2), representing the factors by which to downscale in both directions;
             # Conv2D with the following parameters:
             # filters: 64;
             # kernel_size : (3 x 3);
             # activation : relu;
             # MaxPooling2D with parameter:
             # pool_size : (2,2);
             # Conv2D with the following parameters:
             # filters: 128;
             # kernel_size : (3 x 3);
             # activation : relu;
             # Flatten. This layer Flattens the input. Does not affect the batch size. It is used without parameters;
             # Dense. This layer is a regular fully-connected NN layer. It is used without parameters;
             # units - this is a positive integer, with the meaning: dimensionality of the output space; in this case is: 128;
             # activation - activation function : relu;
             # Dense. This is the final layer (fully connected). It is used with the parameters:
             # units: the number of classes (in our case 10);
             # activation : softmax; for this final layer it is used softmax activation (standard for multiclass classification)
             # Then we compile the model, specifying as well the following parameters:
             # loss;
             # optimizer;
             # metrics.
             # Add convolution 2D
             model_cpc = tf.keras.Sequential()
             # Add convolution 2D
             model_cpc.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu',
                               kernel_initializer='he_normal',
                               input_shape=(IMG_D1, IMG_D2, 1)))
             model_cpc.add(tf.keras.layers.MaxPooling2D((2, 2)))
             # Add dropouts to the model
             model_cpc.add(tf.keras.layers.Dropout(0.25))
             model_cpc.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
             model_cpc.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
             # Add dropouts to the model
             model_cpc.add(tf.keras.layers.Dropout(0.25))
             model_cpc.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
             # Add dropouts to the model
             model_cpc.add(tf.keras.layers.Dropout(0.25))
             model_cpc.add(tf.keras.layers.Flatten())
             model_cpc.add(tf.keras.layers.Dense(128, activation='relu'))
             # Add dropouts to the model
             model_cpc.add(tf.keras.layers.Dropout(0.25))
             #if( (nbC==2) ):
              #   model.add(tf.keras.layers.Dense( (nbC-1), activation='sigmoid'))
             #else: 
             model_cpc.add(tf.keras.layers.Dense( nbC ))
             #if( (nbC==2) ): 
              #   loss_fn = tf.keras.losses.binary_crossentropy
             #else: 
             #
             loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
             model_cpc.compile(loss=loss_fn,optimizer='adam',metrics=['accuracy'])
             #inspect the model
             print( model_cpc.summary() )
            #plot the model
            #You must install pydot (`pip install pydot`) and install graphviz 
            #plot_model(model, to_file='model.png')
            #SVG(model_to_dot(model).create(prog='dot', format='svg'))
             #
             #Run the model
             model_cpc.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=ep, verbose=1)
             #Test prediction accuracy
             ##We calculate the test loss and accuracy.
             #score = model.evaluate(test_images, test_labels, verbose=0)
             #print('Test loss:', score[0])
             #print('Test accuracy:', score[1])
             probability_model_cpc = tf.keras.Sequential([model_cpc,tf.keras.layers.Softmax()])
             cpc_proba_cal.append( probability_model_cpc(X_cal) );
             
             cpc_proba_test.append( probability_model_cpc(train_xdata[test_k[k], :]) )
             
             
             #tf.keras.backend.clear_session()
             #del model_cpc
         return y_cal, cpc_proba_cal, cpc_proba_test
     
    def entropReductionSubSet (p,thr): # attention from 1 to lencth(classes)+1
        #p is the posterior proba, removed is the subset of classes that
        #are put together to reduce entropy
        y = []
        z = []
        for i in range(0, len(p)):
            z.append(p[i])
            y.append(p[i])
        removed=[] #PAJ: changer le nom
        entr_tmp = entropy(y, base=base) #PAJ: verifier le cas de plusieurs valeurs
        i1 = np.argmax(y)
        s=p[i1]
        removed.append(i1+1)
        z[i1] = 0.0
        y.remove(p[i1]) #PAJ: verifier le cas de plusieurs valeurs
        y.append(s)
        while( (len(y)>1) & (entr_tmp>thr) ): #PAJ: a verifier difference entre '&' et 'and'
            #print(entr_tmp)
            i2 = np.argmax(z)
            s+=p[i2]
            removed.append(i2+1)
            y.remove(p[i2])
            y=y[:-1]
            y.append(s)
            entr_tmp = entropy(y, base=base)
            z[i2]=0.0
        return removed 
    
    
    def relabellingEntropy (ytrain_data, entrThr1, entrThr2, posterior_proba, file_posterior, nbC, threshold_nb_classes):#Create a model of ML
        newydata_tmp=[]#new labels: subset in natural numbers
        if file_posterior is not None:
            pp_csv=[]#read posterior probabilities
            with open(file_posterior, "r") as f:
                reader = csv.reader(f, delimiter=",")
                for i, line in enumerate(reader):
                    pp_csv.append( np.array(line,dtype=float) )
            stt=len(pp_csv)-1#the first line is for the classe names
            posterior_proba=pp_csv[1:]

        stt=len(posterior_proba)
        for j in range (stt):
            posterior = posterior_proba[j];
            decOP=int
            ENTROPY=entropy(posterior, base=base)
            if( np.argmax(posterior) == ytrain_data[j] ): # Pourquoi pas un +1 ici ?
                if( ENTROPY>entrThr1):
                    ers=Eclair.entropReductionSubSet (posterior, entrThr1)
                    decOP=Util.b2d( [k in ers for k in range(1,nbC+1)] )
                else:
                    decOP = Util.b2d( [k in [ ytrain_data[j]+1 ] for k in range(1,nbC+1)] )
            else:
                if (ENTROPY>entrThr2 ):
                    ers=Eclair.entropReductionSubSet (posterior, entrThr2)
                    ers.append( ytrain_data[j] + 1)# ytrain_data[j] + 1 could appear two times without risk
                    decOP=Util.b2d( [k in ers for k in range(1,nbC+1)] )
                else:
                    decOP =2**nbC -1 #PAJ il se trompe en etant sur
            newydata_tmp.append( decOP ) 
            
        ### remove and modify some examples 
        nb_nyd=len(newydata_tmp)
        ## relabelling with small occurences
        levels, counts_lvls = np.unique(newydata_tmp, return_counts=True)
        #print('levels before change : '+str(levels))
        nl=len(levels)
        #print('freq levels before change : '+str(counts_lvls))
        #PAJ: threshold_nb_classes is the minimal number of occurency acceptable in a database
        modify=[levels[j] for j in range(nl) if ( counts_lvls[j] < threshold_nb_classes) ]
        
        # newydata_tmp[j] are natural numbers
        for j in range(nb_nyd):
             if( len([item for item in modify if( item==newydata_tmp[j] )])>0 ): #if the example is in modify
                 newydata_tmp[j]=2**ytrain_data[j]#keep the original class
             # if(   ytrain_data[j] not in Util.b2set( Util.d2b(newydata_tmp[j], nbC), nbC )   ):#the original class is not in the relabelled set
             #     remove.append(j)#remove
             #     #add the original class to the relabelledelling
             #     new_SS=Util.b2set( Util.d2b(newydata_tmp[j], nbC), nbC)
             #     relabelled_classes.append( newydata_tmp[j] )
             #     new_SS.append( ytrain_data[j] )
             #     #newydata_tmp[j]=Util.b2d( [t in new_SS for t in range(nbC)] )
             #     original_classes.append( ytrain_data[j] )
        
        levels_f, counts_f = np.unique(newydata_tmp, return_counts=True)   
        #print('levels after changes : '+str(levels_f))
        #print('freq levels after changes : '+str(counts_f))
        
        #PAJ: aucune relabelisation (cas de peu de donnees)!
        if( counts_f[np.argmin(counts_f)] < threshold_nb_classes ):
            newydata_tmp=[]
            print('Do not relabel!')
            for j in range(len(ytrain_data)):
                newydata_tmp.append( 2**ytrain_data[j] )#keep the original class
        
        return newydata_tmp
    
    ## !! Attention conformal prediction produit des empty set comme prediction: de tels exemples sont exclus de l'apprentissage !! ##     
    def relabellingConformalPrediction(train_ydata, cal_y, proba_cal, proba_test, alpha, NB_SPLIT, threshold_nb_classes, nbC):
         partiel_y_train_tmp=[]
             
         for k in range( NB_SPLIT ):
             print("Start measuring non-conformity for calibration")
             n=len(cal_y[k])
             cal_scores=[]
             for i in range(n):
                 cal_scores.append( 1-proba_cal[k][i][cal_y[k][i]] )

             print("Start get adjusted quantile")
             q_level = np.ceil((n+1)*(1-alpha))/n
             qhat = np.quantile(cal_scores, q_level, interpolation='higher')
             print( 'qhat= '+str(qhat))
             
             #prediction for test_xdata
             print("Start predictions")
             prediction_sets = proba_test[k] >= (1-qhat) # 3: form prediction sets
             for i in range( len(prediction_sets) ):
                 #print( prediction_sets[i] )
                 partiel_y_train_tmp.append( Util.b2d( prediction_sets[i].numpy() ) )
                 
             #    
             #    
             #tf.keras.backend.clear_session()
             #del model_cpc
             print(" ... end !")
         
         
         ### remove and modify some examples 
         
         ## empty set relabelling
         nb_nyd=len(partiel_y_train_tmp)
         ## relabelling with small occurences
         levels, counts_lvls = np.unique(partiel_y_train_tmp, return_counts=True)
         print('levels before change : '+str(levels))
         nl=len(levels)
         print('freq levels before change : '+str(counts_lvls))
         modify=[levels[j] for j in range(nl) if ( counts_lvls[j] < threshold_nb_classes) ]
         
         # partiel_y_train_tmp[j] are natural numbers
         original_classes=[]
         relabelled_classes=[]
         remove_es=[]
         remove_fr=[]
         for j in range(nb_nyd):
              if( partiel_y_train_tmp[j] ==0 ):#find element relabelled with empty set
                  remove_es.append(j)#remove
              if( len([item for item in modify if( item==partiel_y_train_tmp[j] )])>0 ): #if the example is in modify
                  partiel_y_train_tmp[j]=2**train_ydata[j]#keep the original class
              if(   (train_ydata[j] in Util.b2set( Util.d2b(partiel_y_train_tmp[j], nbC), nbC ) )==False   ):#the original class is not in the relabelled set
                  remove_fr.append(j)#remove
                  #add the original class to the relabelledelling
                  new_SS=Util.b2set( Util.d2b(partiel_y_train_tmp[j], nbC), nbC)
                  relabelled_classes.append( partiel_y_train_tmp[j] )
                  new_SS.append( train_ydata[j] )
                  #partiel_y_train_tmp[j]=Util.b2d( [t in new_SS for t in range(nbC)] )
                  original_classes.append( train_ydata[j] )
         
            
         #remove elements
         #remove_0=[j for j in range(nb_nyd) if ( partiel_y_train_tmp[j] ==0 ) ]
         remove_all=np.concatenate([remove_es, remove_fr]).tolist()
         print('nb empty set : '+str(len(remove_es)))
         print('nb false reabelling : '+str(len(remove_fr)))
         partiel_y_train=[partiel_y_train_tmp[j] for j in range(nb_nyd) if j not in remove_all ]
         #new_train_ydata=[train_ydata[j] for j in range(nb_nyd) if j not in remove_all ]
            
            
            
         levels_orig, counts_orig = np.unique(original_classes, return_counts=True) 
         print("original classes which are not relebelled with it :"+str(levels_orig)+" with freq. "+str(counts_orig))
         levels_rel, counts_rel = np.unique(relabelled_classes, return_counts=True)                            
         print("relabel instead of true class :"+str(levels_rel)+" with fres. "+str(counts_rel))
         
         levels_f, counts_f = np.unique(partiel_y_train, return_counts=True)   
         print('levels after changes : '+str(levels_f))
         print('freq levels after changes : '+str(counts_f))
         #subsets are represented by their index in the natural order    
         return remove_all, partiel_y_train
     
        
    def prediction_dnn (xtrain_data, ytrain_data, xtest_data, nbC, ep, ts):#ts: test size
        BATCH_SIZE = 128
        IMG_D1=28
        IMG_D2=28
        #TEST_SIZE = 0.2
        RANDOM_STATE = 2018
        #Train the model
        ##Build the model
        ### We will use a Sequential model.
        # Model
        model_masses = tf.keras.Sequential()
 
        # Add convolution 2D
        model_masses.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         kernel_initializer='he_normal',
                         input_shape=(IMG_D1, IMG_D2, 1)))
        model_masses.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model_masses.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Flatten())
        model_masses.add(tf.keras.layers.Dense(128, activation='relu'))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Dense( nbC ))

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_masses.compile(loss=loss_fn,optimizer='adam',metrics=['accuracy'])
        #inspect the model
        #print( model_masses.summary() )
        X_train, X_val, y_train, y_val = train_test_split(xtrain_data, ytrain_data, test_size=ts, random_state=RANDOM_STATE)
        #print("Fashion MNIST train -  rows:",X_train.shape[0]," columns:", X_train.shape[1:4])
        #print("Fashion MNIST valid -  rows:",X_val.shape[0]," columns:", X_val.shape[1:4])
        #print("Fashion MNIST test -  rows:",test_images.shape[0]," columns:", test_images.shape[1:4])
        #Run the model
        model_masses.fit(X_train, y_train,
                          batch_size=BATCH_SIZE,
                          epochs=ep,
                          verbose=1,
                          validation_data=(X_val, y_val))
        probability_model = tf.keras.Sequential([model_masses,tf.keras.layers.Softmax()])
        proba_test = probability_model(xtest_data)
        proba_val = probability_model(X_val)
        #print(probability_model(testData)[:5])
        #tf.keras.backend.clear_session()
        #del model_masses
        return proba_val, y_val, proba_test 
        
    def evidentialPrediction_dnn (self_xdata, newydata, testData, ep):
        levels = np.unique(newydata)
        print(levels)
        BATCH_SIZE = 128
        IMG_D1=28
        IMG_D2=28
        TEST_SIZE = 0.2
        RANDOM_STATE = 2018
        #Train the model
        ##Build the model
        ### We will use a Sequential model.
        # Model
        model_masses = tf.keras.Sequential()
 
        # Add convolution 2D
        model_masses.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         kernel_initializer='he_normal',
                         input_shape=(IMG_D1, IMG_D2, 1)))
        model_masses.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model_masses.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Flatten())
        model_masses.add(tf.keras.layers.Dense(128, activation='relu'))
        # Add dropouts to the model
        model_masses.add(tf.keras.layers.Dropout(0.25))
        model_masses.add(tf.keras.layers.Dense( len(levels) ))

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_masses.compile(loss=loss_fn,optimizer='adam',metrics=['accuracy'])
        #inspect the model
        print( model_masses.summary() )
        series_newydata=pd.Series(newydata, dtype="category")
        #convert subsets, represented by decimals, to numbers from 0:len
        ytrain_masses=series_newydata.map(dict(zip(levels, np.arange(0, len(levels) ))))
        #We further split the train set in train and validation set. 
        #The validation set will be 20% from the original train set, 
        #therefore the split will be train/validation of 0.8/0.2.
        X_train, X_val, y_train, y_val = train_test_split(self_xdata, ytrain_masses, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        #print("Fashion MNIST train -  rows:",X_train.shape[0]," columns:", X_train.shape[1:4])
        #print("Fashion MNIST valid -  rows:",X_val.shape[0]," columns:", X_val.shape[1:4])
        #print("Fashion MNIST test -  rows:",test_images.shape[0]," columns:", test_images.shape[1:4])
        #Run the model
        train_model = model_masses.fit(X_train, y_train,
                          batch_size=BATCH_SIZE,
                          epochs=ep,
                          verbose=1,
                          validation_data=(X_val, y_val))
        probability_model = tf.keras.Sequential([model_masses,tf.keras.layers.Softmax()])
        masses = probability_model(testData);
        #print(probability_model(testData)[:5])
        tf.keras.backend.clear_session()
        del model_masses
        return masses
    
    def evidentialPrediction_nbc (xdata, newydata, testData, cat_feat, num_feat):
        levels = np.unique(newydata)
        #print(levels)
        #Run the model
        mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(xdata,newydata, cat_feat, num_feat, 'gaussian')
        #print(classes)
        u=np.identity(len(classes))
        preds_bo, preds_eu, preds_proba, preds_proba_sm=nbc.predict(testData, cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, u)
        return preds_proba_sm    
                    
    def optimizeAlphaCPForRelabelling(train_images_train, train_labels_train, train_images_val, train_labels_val, cal_y, proba_cal, proba_test):#optimize hyper parameter of relabelling with conformal predicion
        # optimize alpha on train_images_val (initial alpha=0.065) using u65
        perf_u65_PC=[]
        perf_u65_ECLAIR=[]
        perf_u65_SD=[]
        perf_u65_CM=[]
        perf_u65_IC=[]
        alpha_space=np.linspace(0.05, 0.075, num=10)
        for i in range(len(alpha_space)):
            print("\n ---"+str(alpha_space[i]))
            newydata=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=alpha_space[i], NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
            masses=Eclair.evidentialPrediction (np.delete(train_images_train,newydata[0],axis=0), newydata[1], train_images_val, ep=5)
            # print("pignistic Criterion ...")
            # predPC_tst=SVC.pignisticCriterion (masses, np.unique(newydata[1]),10)
            # perf_u65_PC.append( accuracy_score(train_labels_val, predPC_tst, normalize=True, sample_weight=None) )
            print("eclair ... ")
            #optimize eclair hyper-param
            beta=np.linspace(0, 0.8, num=10)
            eclair_beta_u65=[]   
            for j in range(len(beta)):
                predECLAIR=SVC.eclairGFbeta (masses, np.unique(newydata[1]), beta[j], 10)
                acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (train_labels_val,predECLAIR,10)
                eclair_beta_u65.append( u65_ECLAIR )
            j_opt=np.argmax(eclair_beta_u65)    
            perf_u65_ECLAIR.append( [eclair_beta_u65[j_opt], beta[j_opt]] )
            # print("strong Dominance ... ")
            # predSD_tst=SVC.strongDominance (masses, np.unique(newydata[1]),10)
            # acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (train_labels_val,predSD_tst,10)
            # perf_u65_SD.append( u65_SD )
            print("convex Mixture ... ")
            #optimize convex Mixture hyper-param
            lambda_space=np.linspace(0.55, 0.8, num=10)
            CM_lambda_u65=[]
            for j in range(len(lambda_space)):             
                predCM_tst=SVC.convexMixturePrediction (masses, np.unique(newydata[1]),lambda_space[j],10)
                acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (train_labels_val,predCM_tst,10)
                CM_lambda_u65.append(u65_CM )
            j_opt=np.argmax(CM_lambda_u65)  
            perf_u65_CM.append( [CM_lambda_u65[j_opt], lambda_space[j_opt]] )
            # print("interval Criterion ... ")
            # predIC_tst=SVC.intervalCriterion(masses, np.unique(newydata[1]),10)
            # acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (train_labels_val,predIC_tst,10)
            # perf_u65_IC.append( u65_IC )
        #alpha_opt_PC=alpha_space[np.argmax(perf_u65_PC)]    
        opt_eclair=np.argmax( [ perf_u65_ECLAIR[k][0] for k in range(len(alpha_space))] )
        param_opt_ECLAIR= [ alpha_space[opt_eclair], perf_u65_ECLAIR[opt_eclair][1] ]
        #alpha_opt_SD=alpha_space[np.argmax(perf_u65_SD)]
        opt_CM=np.argmax( [ perf_u65_CM[k][0] for k in range(len(alpha_space))] )
        param_opt_CM= [ alpha_space[opt_CM], perf_u65_CM[opt_CM][1] ]        
        #alpha_opt_IC=alpha_space[np.argmax(perf_u65_IC)]

        #print("\n --- alpha_opt is "+str([alpha_opt_PC, param_opt_ECLAIR, alpha_opt_SD, param_opt_CM, alpha_opt_IC]))
        print("\n --- alpha_opt is "+str([param_opt_ECLAIR, param_opt_CM]))
        #return alpha_opt_PC, param_opt_ECLAIR, alpha_opt_SD, param_opt_CM, alpha_opt_IC
        return param_opt_ECLAIR, param_opt_CM
        
    def optimizeThrEntropyRelabelling_dnn(train_images_train, train_labels_train, train_images_val, train_labels_val, proba_test, nbC):#optimize hyper parameter of relabelling with entropy (dnn as prior proba)
        # optimize alpha on train_images_val (initial alpha=0.065) using u65
        perf_u65_PC=[]
        perf_u65_ECLAIR=[]
        perf_u65_SD=[]
        perf_u65_CM=[]
        perf_u65_IC=[]
        thr_entr_space=np.linspace(0.55, 2, num=15)
        for i in range(len(thr_entr_space)):
            print("\n ---"+str(thr_entr_space[i]))
            newydata=Eclair.relabellingEntropy (train_labels_train, thr_entr_space[i], thr_entr_space[i], proba_test, None, nbC, 300)

            masses=Eclair.evidentialPrediction_dnn (train_images_train, newydata, train_images_val, ep=20)
            print("pignistic Criterion ...")


            predPC_tst=SVC.pignisticCriterion (masses, np.unique(newydata),nbC)
            perf_u65_PC.append( accuracy_score(train_labels_val, predPC_tst, normalize=True, sample_weight=None) )
            print("eclair ... ")

            #optimize eclair hyper-param
            beta=np.linspace(0, 1, num=30)
            eclair_beta_u65=[]
            for j in range(len(beta)):
                predECLAIR=SVC.eclairGFbeta (masses, np.unique(newydata), beta[j], nbC)
                acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (train_labels_val,predECLAIR,nbC)
                eclair_beta_u65.append( u65_ECLAIR )
            j_opt=np.argmax(eclair_beta_u65)
            perf_u65_ECLAIR.append( [eclair_beta_u65[j_opt], beta[j_opt]] )
            print("strong Dominance ... ")

            predSD_tst=SVC.strongDominance (masses, np.unique(newydata),nbC)
            acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (train_labels_val,predSD_tst,nbC)
            perf_u65_SD.append( u65_SD )
            print("convex Mixture ... ")

            #optimize convex Mixture hyper-param
            lambda_space=np.linspace(0.55, 0.8, num=20)
            CM_lambda_u65=[]
            for j in range(len(lambda_space)):            
                predCM_tst=SVC.convexMixturePrediction (masses, np.unique(newydata),lambda_space[j],nbC)
                acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (train_labels_val,predCM_tst,nbC)
                CM_lambda_u65.append(u65_CM )
            j_opt=np.argmax(CM_lambda_u65)  
            perf_u65_CM.append( [CM_lambda_u65[j_opt], lambda_space[j_opt]] )
            print("interval Criterion ... ")
            predIC_tst=SVC.intervalCriterion(masses, np.unique(newydata),nbC)
            acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (train_labels_val,predIC_tst,nbC)
            perf_u65_IC.append( u65_IC )

            
        thr_entr_opt_PC=thr_entr_space[np.argmax(perf_u65_PC)]    
        opt_eclair=np.argmax( [ perf_u65_ECLAIR[k][0] for k in range(len(thr_entr_space))] )
        param_opt_ECLAIR= [ thr_entr_space[opt_eclair], perf_u65_ECLAIR[opt_eclair][1] ]
        thr_entr_opt_SD=thr_entr_space[np.argmax(perf_u65_SD)]
        opt_CM=np.argmax( [ perf_u65_CM[k][0] for k in range(len(thr_entr_space))] )
        param_opt_CM= [ thr_entr_space[opt_CM], perf_u65_CM[opt_CM][1] ]
        thr_entr_opt_IC=thr_entr_space[np.argmax(perf_u65_IC)]

        print("\n --- alpha_opt is "+str([thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC]))
        
        return thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC

    
    def optimizeThrEntropyRelabelling_nbc(train_data_train, train_labels_train, train_data_val, train_labels_val, posterior_proba, cat_feat, num_feat, nbC):
        #optimize hyper parameter of relabelling with entropy (dnn as prior proba) 
        perf_u65_PC=[]
        perf_u65_ECLAIR=[]
        perf_u65_SD=[]
        #perf_u65_CM=[]
        #perf_u65_IC=[]
        thr_entr_space=np.linspace(0.2, 3, num=50)
        for i in range(len(thr_entr_space)):
            print("\n ---"+str(thr_entr_space[i]))
            newydata=Eclair.relabellingEntropy (train_labels_train, thr_entr_space[i], thr_entr_space[i], posterior_proba, None, nbC, 20)
            masses=Eclair.evidentialPrediction_nbc (train_data_train, newydata, train_data_val, cat_feat, num_feat)
            #print("pignistic Criterion ...")
            predPC_tst=SVC.pignisticCriterion (masses, np.unique(newydata),nbC)
            perf_u65_PC.append( accuracy_score(train_labels_val, predPC_tst, normalize=True, sample_weight=None) )
            #print("eclair ... ")
            #optimize eclair hyper-param
            beta=np.linspace(0, 3, num=50)
            eclair_beta_u65=[]
            for j in range(len(beta)):
                predECLAIR=SVC.eclairGFbeta (masses, np.unique(newydata), beta[j], nbC)
                acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (train_labels_val,predECLAIR,nbC)
                eclair_beta_u65.append( u65_ECLAIR )
            j_opt=np.argmax(eclair_beta_u65)
            perf_u65_ECLAIR.append( [eclair_beta_u65[j_opt], beta[j_opt]] )
            #print("strong Dominance ... ")
            predSD_tst=SVC.strongDominance (masses, np.unique(newydata),nbC)
            acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (train_labels_val,predSD_tst,nbC)
            perf_u65_SD.append( u65_SD )
            #print("convex Mixture ... ")
            #optimize convex Mixture hyper-param
            # lambda_space=np.linspace(0.35, 0.9, num=20)
            # CM_lambda_u65=[]
            # for j in range(len(lambda_space)):            
            #     predCM_tst=SVC.convexMixturePrediction (masses, np.unique(newydata),lambda_space[j],nbC)
            #     acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (train_labels_val,predCM_tst,nbC)
            #     CM_lambda_u65.append(u65_CM )
            # j_opt=np.argmax(CM_lambda_u65)  
            # perf_u65_CM.append( [CM_lambda_u65[j_opt], lambda_space[j_opt]] )
            # #print("interval Criterion ... ")
            # predIC_tst=SVC.intervalCriterion(masses, np.unique(newydata),nbC)
            # acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (train_labels_val,predIC_tst,nbC)
            # perf_u65_IC.append( u65_IC )
            
        thr_entr_opt_PC=thr_entr_space[np.argmax(perf_u65_PC)]    
        opt_eclair=np.argmax( [ perf_u65_ECLAIR[k][0] for k in range(len(thr_entr_space))] )
        param_opt_ECLAIR= [ thr_entr_space[opt_eclair], perf_u65_ECLAIR[opt_eclair][1] ]
        thr_entr_opt_SD=thr_entr_space[np.argmax(perf_u65_SD)]
        #opt_CM=np.argmax( [ perf_u65_CM[k][0] for k in range(len(thr_entr_space))] )
        #param_opt_CM= [ thr_entr_space[opt_CM], perf_u65_CM[opt_CM][1] ]
        #thr_entr_opt_IC=thr_entr_space[np.argmax(perf_u65_IC)]
        
        #print("\n --- thr_entr_opt is "+str([thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC]))
        print("\n --- thr_entr_opt is "+str([thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD]))
        
        #return thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC
        return thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD
    
    def printRelabelling (f1, self_newydata, f2, self_looProba, f3, self_masses):
        df_nd = pd.DataFrame(self_newydata ) #convert to a dataframe
        df_nd.to_csv(f1,index=False) #save to file
        df_looProba = pd.DataFrame(self_looProba).astype("float")
        df_looProba.to_csv(f2,index=False) #save to file        
        df_masses = pd.DataFrame(self_masses).astype("float")
        df_masses.to_csv(f3,index=False) #save to file
        
    
    def optimization_eclair(nbc):#array 2023
        beta=np.linspace(0, 3, num=50)
        ####################################
        ### optimisation eclair
        beta_perf_eclair, betaStar=Data.optim_beta_eclair(train_images, train_labels, 
                                                          5, 0.5, 0.5, 1200, 10, 5,None, 10)
        ####################################
        #### one to one situations
        arr=[[0.0]*28]*28
        i1=2
        i2=6
        train_images_12=np.array( [arr] )
        train_labels_12=[]
        for i in range( len(train_labels) ):
            if( (train_labels[i]==i1) ): 
                train_images_12=np.append(train_images_12,[train_images[i]], 0)
                train_labels_12.append( 0 )
            if( (train_labels[i]==i2) ): 
                train_images_12=np.append(train_images_12,[train_images[i]], 0)
                train_labels_12.append(1)
        train_images_12=np.delete(train_images_12,0, axis=0)
        train_labels_12=np.array( train_labels_12 )
        test_images_12=np.array( [arr] )
        test_labels_12=[]
        for i in range( len(test_labels) ):
            if( (test_labels[i]==i1)): 
                test_images_12=np.append(test_images_12,[test_images[i] ], 0)
                test_labels_12.append( 0 )
            if( (test_labels[i]==i2)): 
                test_images_12=np.append(test_images_12,[test_images[i] ], 0)
                test_labels_12.append(1)
        test_images_12=np.delete(test_images_12,0, axis=0)
        test_labels_12=np.array( test_labels_12 )
        #
        #optimizing using grid method
        beta_perf_eclair26, betaStar_index_26=Data.optim_beta_eclair(train_images_12, train_labels_12, 
                                                          5, 0.5, 0.5, 600, 10, 5,None, 2)
        fig, ax = plt.subplots()
        ax.set(ylim=[0.75,1])
        cmap = plt.colormaps.get_cmap("tab10")
        colors = cmap.colors  # type: list
        ax.set_prop_cycle(color=colors)
        #plt.xlabel('')
        plt.plot(beta, beta_perf_eclair26, label='validation $u_{65}$ performances')
        #
        text= "beta={:.3f}, u65={:.3f}".format(beta[betaStar_index_26], beta_perf_eclair26[betaStar_index_26])
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                      arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(beta[betaStar_index_26], beta_perf_eclair26[betaStar_index_26]), xytext=(0.94,0.96), **kw)
        ax.hlines(y=beta_perf_eclair26[betaStar_index_26], xmin=0, xmax=3, linestyles='--', linewidth=1, color='gray')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig(dirTMP+'validation_u65_eclair_26.png')
        #result u65 for eclair
        predECLAIR=Data.eclairGFbeta (masses_csv2[1:], selflevels,beta[betaStar_index_26], nbC)
        acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=Data.setValuedClassEvaluation (test_labels,predECLAIR,10)
        #optimizing eclair using method ipmu2022
    
    def optim_beta_eclair(self_xdata, self_ydata, epRelabel, entrThr1, entrThr2, stt, nbRS, epmMsses, file_posterior, nbC):
        #nbC=len(dataFM.classNames)
        BATCH_SIZE = 128
        IMG_D1=28
        IMG_D2=28
        TEST_SIZE = 0.2
        RANDOM_STATE = 2018
        #Train the model
        ##Build the model
        ### We will use a Sequential model.
        # Model
        model_eclair_optim = tf.keras.Sequential()
        #
        # Add convolution 2D
        model_eclair_optim.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         kernel_initializer='he_normal',
                         input_shape=(IMG_D1, IMG_D2, 1)))
        model_eclair_optim.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # Add dropouts to the model
        model_eclair_optim.add(tf.keras.layers.Dropout(0.25))
        model_eclair_optim.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model_eclair_optim.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Add dropouts to the model
        model_eclair_optim.add(tf.keras.layers.Dropout(0.25))
        model_eclair_optim.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        # Add dropouts to the model
        model_eclair_optim.add(tf.keras.layers.Dropout(0.25))
        model_eclair_optim.add(tf.keras.layers.Flatten())
        model_eclair_optim.add(tf.keras.layers.Dense(128, activation='relu'))
        # Add dropouts to the model
        model_eclair_optim.add(tf.keras.layers.Dropout(0.25))
        model_eclair_optim.add(tf.keras.layers.Dense( nbC ))
        #
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_eclair_optim.compile(loss=loss_fn,optimizer='adam',metrics=['accuracy'])
        beta=np.linspace(0, 3, num=50)
        perf_beta=[0]*len(beta)
        for i in range(nbRS):
            print('RANDOM_STATE')
            print(i+1)
            RANDOM_STATE=RANDOM_STATE+1000
            X_train, X_val, y_train, y_val = train_test_split(self_xdata, self_ydata, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            print( type(ytrain[0]) )
            levels, newydata, looProba=Data.relabellingEntropy(X_train, y_train, epRelabel, entrThr1, entrThr2, stt, file_posterior, nbC)
            masses=Data.evidentialPrediction (levels, X_train, X_val, newydata, epmMsses)
            for j in range(len(beta)):
                print('beta')
                print(beta[j])
                pred_eclair_optim=Data.eclairGFbeta (masses, levels, beta[j], nbC)
                acc_eclair_optim, u50_eclair_optim, u65_eclair_optim, u80_eclair_optim, acc_imp_eclair_optim=Data.setValuedClassEvaluation (y_val,pred_eclair_optim,nbC)
                perf_beta[j]=perf_beta[j]+(u65_eclair_optim)/nbRS
            print(perf_beta)
        return perf_beta, np.argmax(perf_beta)


#%% test