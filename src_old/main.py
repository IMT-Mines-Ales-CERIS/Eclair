# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:20:13 2023

@author: imoussaten
"""
#%% imports
import tensorflow as tf
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#print("TensorFlow version:", tf.__version__)
import time
import csv
import numpy as np
import pandas as pd
#import math
#from collections import Counter
import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# from plotly import subplots
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)
#import plotly.figure_factory as ff
#from matplotlib.cm import get_cmap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
#from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy
import random
import itertools
base = 2  # work in units of bits
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import SMOTENC
from collections import Counter
#file from the uncertainty quantification project
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from util import Util
from eclair import Eclair
from setValuedClassification import SVC
from eknn import eknn
from ndc import ndc
from nbc import nbc
from ncc import ncc
from alive_progress import alive_bar

#%% data preprocessing and visualisation
class Data:
    #xdata = []
    #ydata = []
    #classNames = []
    #lenData=0
    #outputs
    #looProba=[]#leave-one-out probabilities
    #newydata = []
    #levels=[] #number of subsets as new labels
    #masses=[]#predicted masses
    #def __init__(self):#, xdata, ydata, classNames):     
        #self.xdata=xdata
        #self.ydata=ydata
        #self.classNames=classNames
        #self.lenData=len(ydata)
    def plotMnist (im, name_im): #im for image and lab for the label of the image
        plt.figure(figsize=(10,10))
        #from matplotlib import pyplot as plt
        plt.imshow(im, cmap=plt.cm.binary)
        plt.xlabel(name_im)
        plt.show()
    def plotMnistGrid (dataToPlotX, names_X, nb, nraw, ncol):
        for i in range(nb):
            plt.subplot(nraw,ncol,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(dataToPlotX[i], cmap=plt.cm.binary)
            plt.xlabel(names_X)
            plt.show()       

    def kNN(x, k, train_data, train_label):
        #create a list of distances between the given image and the images of the training set
        #distances =[np.linalg.norm(x-data[i]) for i in range(len(data))]
        distances =[Data.distance2(x,train_data[i]) for i in range(len(train_data))]
        #Use "np.argpartition". It does not sort the entire array. 
        #It only guarantees that the kth element is in sorted position 
        # and all smaller elements will be moved before it. 
        # Thus the first k elements will be the k-smallest elements.
        idx = np.argpartition(distances, k)
        #print(idx)
        clas, freq = np.unique(train_label[idx[:k]], return_counts=True)
        #print(clas)
        #print(freq)
        return clas[np.argmax(freq)]
    
    def accuracy_kNN(test_data, test_label, train_data, train_label, k):
        cnt = 0
        for x, lab in zip(test_data, test_label):
            if Data.kNN(x,k, train_data, train_label) == lab:
                cnt += 1
        return cnt/len(test_label)                 

    
    def svClassifWithConfPredRelabelling_dnn(xtrain_images, ytrain_images, xtest_images, ytest_images, retrain, optimizeAlpha, trainCP, plotBars, dirTMP):
        if( retrain==True ):
            start_time = time.time()
            #split train data to train and validation (validation is used to optimize parameters)
            train_images_train, train_images_val, train_labels_train, train_labels_val = train_test_split(xtrain_images, ytrain_images, test_size=0.15, random_state=201824) 
            cal_y, proba_cal, proba_test=Eclair.trainModelkFolds_dnn(train_images_train, train_labels_train, ep=10, nbC=10, NB_SPLIT=4)
            print("\n --- %s secondes for train ---" % (time.time() - start_time))

        #### optimize alpha (conformal prediction relabellin)
        if( optimizeAlpha==True ):
            alpha_opt_PC, alpha_opt_ECLAIR, alpha_opt_SD, alpha_opt_CM, alpha_opt_IC = Eclair.optimizeAlphaCPForRelabelling(train_images_train, train_labels_train, train_images_val, train_labels_val, cal_y, proba_cal, proba_test)
        else :
            alpha_opt_PC, alpha_opt_ECLAIR, alpha_opt_SD, alpha_opt_CM, alpha_opt_IC=0.065,0.06111111111111111, 0.065, 0.06111111111111111, 0.0625
            #previous values 0.065,0.055, 0.065, 0.055, 0.0625

        #prediction 
        opt_beta=0.5333333333333333
        opt_lambda=0.5777777777777778
        # relabelling and evidential prediction for alpha==0.065
        if( optimizeAlpha==True ):
            start_time = time.time()
            newydata_PC=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=alpha_opt_PC, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
            print("\n --- %s secondes for relabelling alpha_opt_PC ---" % (time.time() - start_time))
            start_time = time.time()
            masses_PC=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_PC[0],axis=0), newydata_PC[1], xtest_images, ep=10)
            print("\n --- %s secondes for evidential Predictions alpha_opt_PC ---" % (time.time() - start_time) )
            start_time = time.time()
            newydata_eclair=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=alpha_opt_ECLAIR, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
            print("\n --- %s secondes for relabelling eclair ---" % (time.time() - start_time))
            start_time = time.time()
            masses_eclair=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_eclair[0],axis=0), newydata_eclair[1], xtest_images, ep=10)
            print("\n --- %s secondes for evidential Predictions eclair ---" % (time.time() - start_time) )
            start_time = time.time()
            newydata_SD=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=alpha_opt_SD, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
            print("\n --- %s secondes for relabelling SD ---" % (time.time() - start_time))
            start_time = time.time()
            masses_SD=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_SD[0],axis=0), newydata_SD[1], xtest_images, ep=10)
            print("\n --- %s secondes for evidential Predictions SD ---" % (time.time() - start_time) )
            start_time = time.time()
            newydata_CM=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=alpha_opt_CM, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
            print("\n --- %s secondes for relabelling CM ---" % (time.time() - start_time))
            start_time = time.time()
            masses_CM=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_CM[0],axis=0), newydata_CM[1], xtest_images, ep=10)
            print("\n --- %s secondes for evidential Predictions CM ---" % (time.time() - start_time) )
            start_time = time.time()
            newydata_IC=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=alpha_opt_IC, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
            print("\n --- %s secondes for relabelling SD ---" % (time.time() - start_time))
            start_time = time.time()
            masses_IC=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_IC[0],axis=0), newydata_IC[1], xtest_images, ep=10)
            print("\n --- %s secondes for evidential Predictions SD ---" % (time.time() - start_time) )
        else :
            if( retrain==True ):
                start_time = time.time()
                newydata_065=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=0.065, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
                print("\n --- %s secondes for relabelling 0.065 ---" % (time.time() - start_time))
                start_time = time.time()
                masses_065=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_065[0],axis=0), newydata_065[1], xtest_images, ep=10)
                print("\n --- %s secondes for evidential Predictions 0.065 ---" % (time.time() - start_time) )
        
                # relabelling and evidential prediction for alpha==0.055
                start_time = time.time()
                newydata_055=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=0.055, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
                print("\n --- %s secondes for relabelling 0.055 ---" % (time.time() - start_time))
                start_time = time.time()
                masses_055=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_055[0],axis=0), newydata_055[1], xtest_images, ep=10)
                print("\n --- %s secondes for evidential Predictions 0.055 ---" % (time.time() - start_time) )
        
                # relabelling and evidential prediction for alpha==0.0625
                start_time = time.time()
                newydata_0625=Eclair.relabellingConformalPrediction(train_labels_train, cal_y, proba_cal, proba_test, alpha=0.0625, NB_SPLIT=4, threshold_nb_classes=300, nbC=10)
                print("\n --- %s secondes for relabelling 0.0625 ---" % (time.time() - start_time))
                start_time = time.time()
                masses_0625=Eclair.evidentialPrediction_dnn (np.delete(train_images_train,newydata_0625[0],axis=0), newydata_0625[1], xtest_images, ep=10)
                print("\n --- %s secondes for evidential Predictions 0.0625 ---" % (time.time() - start_time) )

        #prediction
        perf_u=[]
        if( optimizeAlpha==True ):
            start_time = time.time()
            predPC_tst=SVC.pignisticCriterion (masses_065, np.unique(newydata_065[1]),10)
            acc_PC=accuracy_score(ytest_images, predPC_tst, normalize=True, sample_weight=None)
            print("\n --- %s secondes for PC ---" % np.round( (time.time() - start_time),3) )
            perf_u.append( [acc_PC, acc_PC, acc_PC, acc_PC] )
    
            start_time = time.time()
            predECLAIR=SVC.eclairGFbeta (masses_055, np.unique(newydata_055[1]), opt_beta, 10)
            acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (ytest_images,predECLAIR,10)
            print( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR])
            print("\n --- %s secondes for Eclair ---" % np.round( (time.time() - start_time),3) )
            perf_u.append( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR] )
    
            start_time = time.time()
            predSD_tst=SVC.strongDominance (masses_065, np.unique(newydata_065[1]),10)
            acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (ytest_images,predSD_tst,10)
            print( [acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
            print("\n --- %s secondes for SD ---" % np.round( (time.time() - start_time),3) )
            perf_u.append([acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
    
            start_time = time.time()
            predCM_tst=SVC.convexMixturePrediction (masses_055, np.unique(newydata_055[1]),opt_lambda,10)
            acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (ytest_images,predCM_tst,10)
            print( [acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
            print("\n --- %s secondes for CM ---" % np.round( (time.time() - start_time),3) )
            perf_u.append([acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
    
            start_time = time.time()
            predIC_tst=SVC.intervalCriterion(masses_0625, np.unique(newydata_0625[1]),10)
            acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (ytest_images,predIC_tst,10)
            print( [acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
            print("\n --- %s secondes for IC ---" % np.round( (time.time() - start_time),3) )
            perf_u.append([acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
        else :
            if( retrain==True ):
                start_time = time.time()
                predPC_tst=SVC.pignisticCriterion (masses_PC, np.unique(newydata_PC[1]),10)
                acc_PC=accuracy_score(ytest_images, predPC_tst, normalize=True, sample_weight=None)
                print("\n --- %s secondes for PC ---" % np.round( (time.time() - start_time),3) )
                perf_u.append( [acc_PC, acc_PC, acc_PC, acc_PC, acc_PC] )
                start_time = time.time()
                predECLAIR=SVC.eclairGFbeta (masses_eclair, np.unique(newydata_eclair[1]), opt_beta, 10)
                acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (ytest_images,predECLAIR,10)
                print( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR])
                print("\n --- %s secondes for Eclair ---" % np.round( (time.time() - start_time),3) )
                perf_u.append( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR] )
                start_time = time.time()
                predSD_tst=SVC.strongDominance (masses_SD, np.unique(newydata_SD[1]),10)
                acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (ytest_images,predSD_tst,10)
                print( [acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
                print("\n --- %s secondes for SD ---" % np.round( (time.time() - start_time),3) )
                perf_u.append([acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
                start_time = time.time()
                predCM_tst=SVC.convexMixturePrediction (masses_CM, np.unique(newydata_CM[1]),opt_lambda,10)
                acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (ytest_images,predCM_tst,10)
                print( [acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
                print("\n --- %s secondes for CM ---" % np.round( (time.time() - start_time),3) )
                perf_u.append([acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
                start_time = time.time()
                predIC_tst=SVC.intervalCriterion(masses_IC, np.unique(newydata_IC[1]),10)
                acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (ytest_images,predIC_tst,10)
                print( [acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
                print("\n --- %s secondes for IC ---" % np.round( (time.time() - start_time),3) )
                perf_u.append([acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
            else :
                perf_u.append( [0.8897]*5 )
                perf_u.append( [0.849, 0.8832, 0.89346, 0.9037200000000001, 0.9174] )
                perf_u.append( [0.8425, 0.8831950000000001, 0.89552085, 0.9078467, 0.9251] )
                perf_u.append( [0.8566, 0.88495, 0.893475, 0.902, 0.9135] )
                perf_u.append( [0.8526, 0.8864333333333332, 0.8966166666666666, 0.9068000000000002, 0.9206] )
                
                
            
        if(  trainCP==True ):
            start_time = time.time()
            predCP_tst=SVC.conformalPrediction(train_images, train_labels, test_images, 0.065, 10, 10)
            acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP=SVC.setValuedClassEvaluation (test_labels,predCP_tst,10)
            print( [acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
            print("\n --- %s secondes for CP ---" % np.round( (time.time() - start_time),3) ) 
            perf_u.append([acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
        else :#performances with alpha=0.065
            perf_u.append([0.8652, 0.89805, 0.9079050000000001, 0.9177600000000002, 0.9309])
            
        print(perf_u)
        if( plotBars==True ):
            Data.barPlotPredictions(barsData=[np.array(perf_u[0]) *100, np.array(perf_u[1]) *100, np.array(perf_u[2]) *100, np.array(perf_u[3]) *100, np.array(perf_u[4]) *100, np.array(perf_u[5]) *100], 
                                    maxData=[100]*5,
                                    ytitle='Set-valued Classif', barWidth=0.1, h_lim=[-0.5,6.5], 
                                    errorData=[[0]*5, [0]*5, [0]*5, [0]*5, [0]*5, [0]*5],
                                   classifierNames =['PC', 'ECLAIR', 'SD','$CM_{0.7}$', '2pic', 'Conformal P.'], 
                                   groupNames=['acc', 'u50', 'u65', 'u80', 'imp'],
                                   saveFile=dirTMP+'testallCl.png', y_lim=[65, 101])
        return perf_u
    
    def svClassifWithEntropyRelabelling_dnn(xtrain_images, ytrain_images, xtest_images, ytest_images, plotBars, dirTMP, nbC, ep_test):
        
        # xtrain_images=train_images
        # ytrain_images=train_labels
        # xtest_images=test_images 
        # ytest_images=test_labels
        #ep_test=50#ipmu2024
        #nbC=10
        start_time = time.time()
        #split train data to train and validation (validation is used to optimize parameters)
        train_images_train, train_images_val, train_labels_train, train_labels_val = train_test_split(xtrain_images, ytrain_images, test_size=0.15, random_state=201824) 
        NB_SPLIT=4
        cal_y, proba_cal, proba_test=Eclair.trainModelkFolds_dnn(train_images_train, train_labels_train, ep=ep_test, nbC=nbC, NB_SPLIT=NB_SPLIT)
        print("\n --- %s secondes for train ---" % (time.time() - start_time))
        posterior_proba=np.concatenate( [proba_test[j] for j in range(NB_SPLIT)])
        
        #thr_entr_opt_PC, params_opt_ECLAIR, thr_entr_opt_SD, params_opt_CM, thr_entr_opt_IC=Eclair.optimizeThrEntropyRelabelling_dnn(train_images_train, train_labels_train, train_images_val, train_labels_val, posterior_proba, 10)
        
        # opt_thr_entr=[1.7410714285714286, 1.275, 1.6892857142857143, 1.7410714285714286, 1.3785714285714286]#[PC, Eclair, SD, CM, IC]
        # opt_beta_eclair=0.3448275862068966
        # opt_lambda_CM=0.7473684210526317
        opt_thr_entr=[1.1714285714285715, 1.0678571428571428, 1.0678571428571428, 1.0678571428571428, 1.0678571428571428]#ep50 [PC, Eclair, SD, CM, IC]
        opt_beta_eclair=0.4482758620689655#beta eclair ep50
        opt_lambda_CM=0.55#lambda CM ep50
                    
        #relabelling
        print("\n ---relabelling")
        start_time = time.time()
        newydata_PC=Eclair.relabellingEntropy (train_labels_train, entrThr1=opt_thr_entr[0], entrThr2=opt_thr_entr[0], posterior_proba=posterior_proba, file_posterior=None, nbC=nbC, threshold_nb_classes=300)
        print("\n --- %s secondes for relabelling alpha_opt_PC ---" % (time.time() - start_time))
        start_time = time.time()
        masses_PC=Eclair.evidentialPrediction_dnn (train_images_train, newydata_PC, xtest_images, ep=ep_test)
        print("\n --- %s secondes for evidential Predictions alpha_opt_PC ---" % (time.time() - start_time) )
        start_time = time.time()
        newydata_eclair=Eclair.relabellingEntropy (train_labels_train, entrThr1=opt_thr_entr[1], entrThr2=opt_thr_entr[1], posterior_proba=posterior_proba, file_posterior=None, nbC=nbC, threshold_nb_classes=300)
        print("\n --- %s secondes for relabelling eclair ---" % (time.time() - start_time))
        start_time = time.time()
        masses_eclair=Eclair.evidentialPrediction_dnn (train_images_train, newydata_eclair, xtest_images, ep=ep_test)
        print("\n --- %s secondes for evidential Predictions eclair ---" % (time.time() - start_time) )
        start_time = time.time()
        newydata_SD=Eclair.relabellingEntropy (train_labels_train, entrThr1=opt_thr_entr[2], entrThr2=opt_thr_entr[2], posterior_proba=posterior_proba, file_posterior=None, nbC=nbC, threshold_nb_classes=300)
        print("\n --- %s secondes for relabelling SD ---" % (time.time() - start_time))
        start_time = time.time()
        masses_SD=Eclair.evidentialPrediction_dnn (train_images_train, newydata_SD, xtest_images, ep=ep_test)
        print("\n --- %s secondes for evidential Predictions SD ---" % (time.time() - start_time) )
        start_time = time.time()
        newydata_CM=Eclair.relabellingEntropy (train_labels_train, entrThr1=opt_thr_entr[3], entrThr2=opt_thr_entr[3], posterior_proba=posterior_proba, file_posterior=None, nbC=nbC, threshold_nb_classes=300)
        print("\n --- %s secondes for relabelling CM ---" % (time.time() - start_time))
        start_time = time.time()
        masses_CM=Eclair.evidentialPrediction_dnn (train_images_train, newydata_CM, xtest_images, ep=ep_test)
        print("\n --- %s secondes for evidential Predictions CM ---" % (time.time() - start_time) )
        start_time = time.time()
        newydata_IC=Eclair.relabellingEntropy (train_labels_train, entrThr1=opt_thr_entr[4], entrThr2=opt_thr_entr[4], posterior_proba=posterior_proba, file_posterior=None, nbC=nbC, threshold_nb_classes=300)
        print("\n --- %s secondes for relabelling SD ---" % (time.time() - start_time))
        start_time = time.time()
        masses_IC=Eclair.evidentialPrediction_dnn (train_images_train, newydata_IC, xtest_images, ep=ep_test)
        print("\n --- %s secondes for evidential Predictions SD ---" % (time.time() - start_time) )
        
        #Predictions
        print("\n ---predictions")
        perf_u_ep50=[]
        start_time = time.time()
        predPC_tst=SVC.pignisticCriterion (masses_PC, np.unique(newydata_PC),nbC)
        #acc_PC=accuracy_score(ytest_images, predPC_tst, normalize=True, sample_weight=None)
        acc_PC, u50_PC, u65_PC, u80_PC, acc_imp_PC=SVC.setValuedClassEvaluation (ytest_images,predPC_tst,nbC)
        print( [acc_PC, u50_PC, u65_PC, u80_PC, acc_imp_PC] )
        print("\n --- %s secondes for PC ---" % np.round( (time.time() - start_time),3) )
        perf_u_ep50.append( [acc_PC, acc_PC, acc_PC, acc_PC, acc_PC] )
        start_time = time.time()
        predECLAIR=SVC.eclairGFbeta (masses_eclair, np.unique(newydata_eclair), opt_beta_eclair, nbC)
        acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (ytest_images,predECLAIR,nbC)
        print( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR])
        print("\n --- %s secondes for Eclair ---" % np.round( (time.time() - start_time),3) )
        perf_u_ep50.append( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR] )
        start_time = time.time()
        predSD_tst=SVC.strongDominance (masses_SD, np.unique(newydata_SD),nbC)
        acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (ytest_images,predSD_tst,nbC)
        print( [acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
        print("\n --- %s secondes for SD ---" % np.round( (time.time() - start_time),3) )
        perf_u_ep50.append([acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
        start_time = time.time()
        predWD_tst=SVC.weakDominance (masses_SD, np.unique(newydata_SD),nbC)
        acc_WD, u50_WD, u65_WD, u80_WD, acc_imp_WD=SVC.setValuedClassEvaluation (ytest_images,predWD_tst,nbC)
        print( [acc_WD, u50_WD, u65_WD, u80_WD, acc_imp_WD])
        print("\n --- %s secondes for WD ---" % np.round( (time.time() - start_time),3) )
        perf_u_ep50.append([acc_WD, u50_WD, u65_WD, u80_WD, acc_imp_WD])
        start_time = time.time()
        predCM_tst=SVC.convexMixturePrediction (masses_CM, np.unique(newydata_CM),opt_lambda_CM,nbC)
        acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (ytest_images,predCM_tst,nbC)
        print( [acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
        print("\n --- %s secondes for CM ---" % np.round( (time.time() - start_time),3) )
        perf_u_ep50.append([acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
        start_time = time.time()
        predIC_tst=SVC.intervalCriterion(masses_IC, np.unique(newydata_IC),nbC)
        acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (ytest_images,predIC_tst,nbC)
        print( [acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
        print("\n --- %s secondes for IC ---" % np.round( (time.time() - start_time),3) )
        perf_u_ep50.append([acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
        
        
        #Generalized criterion with the same parameters as eclair :A revoir !
        # start_time = time.time()
        
        # predECLAIR=SVC.predection_genCriteria (masses_eclair, np.unique(newydata_eclair),u=np.identity(nbC), 
        #                                        criterion=[True,True,True], gamma_all=0.8, alpha_fhc=0.6, beta_gowac=0.6)
        # acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (ytest_images,predECLAIR,nbC)
        # print( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR])
        # print("\n --- %s secondes for Eclair ---" % np.round( (time.time() - start_time),3) )
        
        
        #predictions NDC
        print("\n ---NDC")
        #perf_u65, beta_optim=ndc.optim_beta_dnn(train_images_train, train_labels_train, train_images_val, train_labels_val, nbC, ep=ep_test)
        beta_optim=1.263157894736842 # for ep_test=50
        post_proba=ndc.posteriorTest_dnn(xtrain_images, ytrain_images, xtest_images, nbC, ep=ep_test)
        pred_ndc=ndc.predict(post_proba, beta_optim)
        acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc=SVC.setValuedClassEvaluation (ytest_images,pred_ndc,nbC)
        print( [acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc])
        perf_u_ep50.append([acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc])
        
        
        #conformal
        print("\n ---Conformal prediction")
        #train_images_train, train_images_val, train_labels_train, train_labels_val
        #xtrain_images, ytrain_images, xtest_images, ytest_images
        proba_cal_val, y_cal_val, proba_val=Eclair.prediction_dnn (train_images_train, train_labels_train, train_images_val, nbC, ep=ep_test, ts=0.2)#ts: test size
        perf_u65, alpha_optim=SVC.optimizeAlpha_cp(y_cal_val, proba_cal_val, proba_val, train_labels_val, nbC)
        proba_cal_test, y_cal_test, proba_test=Eclair.prediction_dnn (xtrain_images, ytrain_images, xtest_images, nbC, ep=ep_test, ts=0.2)#ts: test size
        predCP_tst=SVC.conformalPrediction(y_cal_test, proba_cal_test,  proba_test, alpha_optim)
        acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP=SVC.setValuedClassEvaluation (ytest_images,predCP_tst,nbC)
        perf_u_ep50.append([acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
        print( [acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
        
        perf_u_ep10=[
            [0.8998, 0.8998, 0.8998, 0.8998, 0.8998], #PC
            [0.8851, 0.89602, 0.8993008, 0.9025816000000002, 0.9071], #Eclair
            [0.8644, 0.8846352380952381, 0.8915776653061225, 0.8985200925170069, 0.9289], #SD
            [0.8965, 0.8965, 0.8965, 0.8965000000000002, 0.8965], #WD
            [0.8909, 0.8991033333333333, 0.9016399666666666, 0.9041766000000001, 0.9082], #CM
            [0.8675, 0.8924761904761904, 0.900451331292517, 0.9084264721088435, 0.9264], #IC
            [0.878, 0.9040666666666666, 0.9119608333333333, 0.9198550000000001, 0.9309], #NDC
            [0.8475, 0.8923833333333334, 0.9058816666666667, 0.9193800000000001, 0.9376] #Conformal
            ]
        perf_u_ep50=[
                [0.912, 0.912, 0.912, 0.912, 0.912],#PC
                 [0.9086, 0.90989, 0.9105866, 0.9112832000000002, 0.9215],#Eclair
                 [0.8738, 0.8881842857142858, 0.8936569326530612, 0.8991295795918368, 0.9398], #SD
                 [0.9071, 0.9071, 0.9071, 0.9071000000000001, 0.9071],#WD
                 [0.9002, 0.9070476190476191, 0.9093857993197278, 0.9117239795918369, 0.9224],#CM
                 [0.8874, 0.9013716666666667, 0.9063933833333333, 0.9114151000000001, 0.9405],#IC
                 [0.8778, 0.9152833333333334, 0.9266966666666667, 0.93811, 0.9545],#NDC
                 [0.895, 0.91895, 0.926135, 0.93332, 0.9429]#Conformal
                 ]
        
        perf_u_ep50_new_optim=[
                [0.9168, 0.9168, 0.9168, 0.9168, 0.9168],#PC
         [0.8984, 0.90656, 0.9092264000000001, 0.9118928000000002, 0.922],#Eclair
         [0.8786, 0.8984233333333334, 0.9050413333333334, 0.9116593333333334, 0.9362],#SD
         [0.9077, 0.9089, 0.9092599999999998, 0.9096200000000001, 0.9101],#WD
         [0.8949, 0.9074949999999999, 0.91153435, 0.9155737, 0.9264],#CM
         [0.8831, 0.9013483333333333, 0.9074793166666666, 0.9136103000000001, 0.9384],#IC
         [0.8808, 0.91665, 0.9275950000000001, 0.9385400000000002, 0.9545],#NDC
         [0.8859, 0.9162333333333333, 0.9253366666666667, 0.9344400000000002, 0.9466]#Conformal
         ]
        
        #if( plotBars==True ):
        perf_u=perf_u_ep50_new_optim
        Data.barPlotPredictions(barsData=[np.array(perf_u[0]) *100, np.array(perf_u[1]) *100, np.array(perf_u[2]) *100, np.array(perf_u[3]) *100, 
                                          np.array(perf_u[4]) *100, np.array(perf_u[5]) *100, 
                                              np.array(perf_u[6]) *100, np.array(perf_u[7]) *100], 
                                    maxData=[100]*5,
                                    ytitle='u', barWidth=0.1, h_lim=[-0.1,4.8], 
                                    errorData=[[0]*5, [0]*5, [0]*5, [0]*5, [0]*5, [0]*5, [0]*5, [0]*5],
                                   classifierNames =['PC', 'GFB', 'SD', 'WD', 'CM','2pic', 'NDC', 'Conformal'], 
                                   groupNames=['acc', 'u50', 'u65', 'u80', 'imp'],
                                   saveFile=dirTMP+'CM_ipmu2024/perf_fmData_ci_01mars.png', y_lim=[80, 101])
        return perf_u
    
    def perfSVClassif_UCI_data(uci_Xtrain, uci_ytrain, train_data_train, train_data_val, train_labels_train, train_labels_val, uci_Xtest, uci_ytest, cat_feat, num_feat, posterior_proba, thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC, rand_st, nbC):
        perf_u=[]

            
        mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(train_data_train,train_labels_train, cat_feat, num_feat, 'gaussian')
        u=np.identity(len(classes))
        preds_bo, preds_eu, preds_proba, preds_proba_sm=nbc.predict(uci_Xtest, cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, u)
        acc_nbc=accuracy_score(uci_ytest, preds_bo, normalize=True, sample_weight=None)
        perf_u.append( [acc_nbc]*5 )
        
        #plt.cla()    
        newydata=Eclair.relabellingEntropy (train_labels_train, thr_entr_opt_PC, thr_entr_opt_PC, posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(train_data_train, newydata, uci_Xtest, cat_feat, num_feat)
        # plt.subplot(3,2,1)
        # plt.grid(False)
        # for i_m in range(len(masses[0,:])):
        #     plt.plot(masses[:,i_m], '.', label=np.unique(newydata)[i_m])
        # plt.xlabel("PC")
        # plt.legend()
        # plt.show() 
        #print( np.unique(newydata, return_counts=True))
        predPC_tst=SVC.pignisticCriterion (masses, np.unique(newydata),nbC)
        acc_PC=accuracy_score(uci_ytest, predPC_tst, normalize=True, sample_weight=None)
        perf_u.append( [acc_PC]*5 )
        
        newydata=Eclair.relabellingEntropy (train_labels_train, param_opt_ECLAIR[0], param_opt_ECLAIR[0], posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(train_data_train, newydata, uci_Xtest, cat_feat, num_feat)
        #print(masses)
        # plt.subplot(3,2,2)
        # plt.grid(False)
        # for i_m in range(len(masses[0,:])):
        #     plt.plot(masses[:,i_m], '.', label=np.unique(newydata)[i_m])
        # plt.xlabel("Eclair")
        # plt.legend()
        # plt.show()
        #print( np.unique(newydata, return_counts=True))
        predECLAIR=SVC.eclairGFbeta (masses, np.unique(newydata), param_opt_ECLAIR[1], nbC)
        acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (uci_ytest,predECLAIR,nbC)
        #print( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR])
        perf_u.append( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR] )
        
        newydata=Eclair.relabellingEntropy (train_labels_train, thr_entr_opt_SD, thr_entr_opt_SD, posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(train_data_train, newydata, uci_Xtest, cat_feat, num_feat)
        # plt.subplot(3,2,3)
        # plt.grid(False)
        # for i_m in range(len(masses[0,:])):
        #     plt.plot(masses[:,i_m], '.', label=np.unique(newydata)[i_m])
        # plt.xlabel("SD")
        # plt.legend()
        # plt.show()
        #print( np.unique(newydata, return_counts=True))
        predSD_tst=SVC.strongDominance (masses, np.unique(newydata),nbC)
        acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (uci_ytest,predSD_tst,nbC)
        #print( [acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
        perf_u.append([acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
        
        newydata=Eclair.relabellingEntropy (train_labels_train, param_opt_CM[0], param_opt_CM[0], posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(train_data_train, newydata, uci_Xtest, cat_feat, num_feat)
        # plt.subplot(3,2,4)
        # plt.grid(False)
        # for i_m in range(len(masses[0,:])):
        #     plt.plot(masses[:,i_m], '.', label=np.unique(newydata)[i_m])
        # plt.xlabel("MC")
        # plt.legend()
        # plt.show()
        #print( np.unique(newydata, return_counts=True))
        predCM_tst=SVC.convexMixturePrediction (masses, np.unique(newydata),param_opt_CM[1],nbC)
        acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (uci_ytest,predCM_tst,nbC)
        #print( [acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
        perf_u.append([acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM])
        
        newydata=Eclair.relabellingEntropy (train_labels_train, thr_entr_opt_IC, thr_entr_opt_IC, posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(train_data_train, newydata, uci_Xtest, cat_feat, num_feat)
        # plt.subplot(3,2,5)
        # plt.grid(False)
        # for i_m in range(len(masses[0,:])):
        #     plt.plot(masses[:,i_m], '.', label=np.unique(newydata)[i_m])
        # plt.xlabel("IC")
        # plt.legend()
        # plt.show()
        #print( np.unique(newydata, return_counts=True))
        predIC_tst=SVC.intervalCriterion(masses, np.unique(newydata),nbC)
        acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (uci_ytest,predIC_tst,nbC)
        #print( [acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
        perf_u.append([acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC])
        
        
        
        #conformal prediction
        X_train, X_cal, y_train, y_cal = train_test_split(train_data_train, train_labels_train, test_size=0.2, random_state=2018)
        mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(X_train, y_train, cat_feat, num_feat, 'gaussian')
        u=np.identity(len(classes))
        preds_bo_cal, preds_eu_cal, preds_proba_cal, preds_proba_sm_cal=nbc.predict(X_cal, cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, u)
        preds_bo_val, preds_eu_val, preds_proba_val, preds_proba_sm_val=nbc.predict(train_data_val, cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, u)
        preds_bo_test, preds_eu_test, preds_proba_test, preds_proba_sm_test=nbc.predict(uci_Xtest, cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, u)
        # plt.subplot(3,2,6)
        # plt.grid(False)
        # for i_m in range(len(preds_proba_sm[0,:])):
        #     plt.plot(preds_proba_sm[:,i_m], '.', label=classes[i_m])
        # plt.xlabel("CP")
        # plt.legend()
        # plt.show()
        perf_u65, alpha_optim=SVC.optimizeAlpha_cp_nbc(y_cal, preds_proba_sm_cal, preds_proba_sm_val, train_labels_val, nbC)
        predCP_tst=SVC.conformalPrediction_nbc(y_cal, preds_proba_sm_cal,  preds_proba_sm_test, alpha_optim)
        acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP=SVC.setValuedClassEvaluation (uci_ytest,predCP_tst,nbC)
        #print( [acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
        perf_u.append([acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
        
        #predictions NDC
        #train_data_train, train_data_val, train_labels_train, train_labels_val = train_test_split(data_uci_Xtrain[0], data_uci_ytrain[0], test_size=0.15, random_state=201824)
        #nbC=len(np.unique(data_uci_ytrain[0]))
        perf_u65, beta_optim=ndc.optim_beta_nbc(train_data_train, train_labels_train, train_data_val, train_labels_val, cat_feat, num_feat, nbC)
            
        posterior_proba=ndc.posteriorTest_nbc(train_data_train, train_labels_train, cat_feat, num_feat, uci_Xtest)
        pred_ndc=ndc.predict(posterior_proba, beta_optim)
        acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc=SVC.setValuedClassEvaluation (uci_ytest,pred_ndc,nbC)
        #print( [acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc])
        perf_u.append([acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc])
        
        
        #ncc predictions
        cpts=Util.points_discretization(uci_Xtrain, uci_ytrain, num_feat)
        #print(cpts)
        x_train_disc=Util.discretization(uci_Xtrain, cpts, num_feat )
        x_test_disc=Util.discretization(uci_Xtest, cpts,num_feat )
        train_data_train_ncc, train_data_val_ncc, train_labels_train_ncc, train_labels_val_ncc = train_test_split(x_train_disc, uci_ytrain, test_size=0.15, random_state=201824)
        #nbC=len(np.unique(uci_ytrain))
        N, classes, nc, p_c, lev, freq, p_f=ncc.modelNCC(train_data_train_ncc, train_labels_train_ncc)
        ncc_perf_u65, s_opt=ncc.optimize_s(train_data_val_ncc, train_labels_val_ncc, classes, nc, lev, freq)
        preds_ncc=ncc.predict(x_test_disc, s_opt, classes, nc, lev, freq)
        acc_ncc, u50_ncc, u65_ncc, u80_ncc, acc_imp_ncc=SVC.setValuedClassEvaluation (uci_ytest,preds_ncc,nbC)
        #print( [acc_ncc, u50_ncc, u65_ncc, u80_ncc, acc_imp_ncc])
        perf_u.append([acc_ncc, u50_ncc, u65_ncc, u80_ncc, acc_imp_ncc])
        

        
        return perf_u
        
    def svClassifWithEntropyRelabelling_nbc_uci(data_uci_names, data_uci_classes, data_uci_Xtrain, data_uci_Xtest, data_uci_ytrain, data_uci_ytest, cat_features, num_features, OptimzeParams):
        params=[]
        params_optim=[
                [1.0, [1.0, 0.0], 1.0, [1.0, 0.6973684210526316], 1.0],
                [0.2, [0.2, 0.0], 0.2, [0.2, 0.35], 0.2],
                [1.0571428571428572, [1.0571428571428572, 0.0], 0.2, [0.2, 0.35], 0.2],
                [0.2, [0.2, 0.0], 0.2, [0.2, 0.35], 0.2],
                [0.9428571428571428,[0.9428571428571428, 0.0], 0.9428571428571428, [0.9428571428571428, 0.35], 0.9428571428571428],
                [0.2, [0.2, 0.0], 0.2, [0.2, 0.35], 0.2],
                [1.0, [1.0, 0.0], 1.0, [1.0, 0.35], 1.0],
                [0.2, [0.2, 0.0], 0.2, [0.2, 0.35], 0.2],
                [0.6, [0.6, 0.061224489795918366], 0.6, [0.6, 0.35], 0.6],
                [1.0571428571428572, [0.2, 0.0], 0.2, [0.2, 0.35], 0.2],
                [1.0571428571428572, [1.0571428571428572, 0.0], 1.0571428571428572, [1.0571428571428572, 0.35], 1.0571428571428572],
                [0.2, [1.6285714285714286, 0.0], 1.6285714285714286, [1.6285714285714286, 0.35], 1.6285714285714286]
                ]
        perf_all=[]
        classifiers=['NBC', 'PC', 'ECLAIR', 'SD','CM', '2pic', 'CP', 'NCC', 'NDC']
        metrics=['acc', 'u50', 'u65', 'u80', 'imp']
        for data_index in range( len(data_uci_names)):
            print(data_uci_names[data_index])
            perf_all.append( [] )
            for i_c in range( len(classifiers) ):
                perf_all[data_index].append( [] )
            for k in range(50):
                print('k = '+str(k))
                #split train data to train and validation (validation is used to optimize parameters)
                train_data_train, train_data_val, train_labels_train, train_labels_val = train_test_split(data_uci_Xtrain[data_index], data_uci_ytrain[data_index], test_size=0.15, random_state=2018+(data_index*k) )
                #print('sampled dataset shape %s' % Counter(train_labels_train))
                if( len(cat_features[data_index])==0 ):
                    sm = SMOTE(random_state=2018, k_neighbors=2)
                else :
                    sm = SMOTENC(random_state=42, categorical_features=cat_features[data_index])
                train_data_train, train_labels_train = sm.fit_resample(train_data_train, train_labels_train)
                #print('Resampled dataset shape %s' % Counter(train_labels_train))
                NB_SPLIT=4
                nbC=len(np.unique(data_uci_ytrain[data_index]))
                y_cal, proba_cal, proba_test=Eclair.trainModelkFolds_nbc(train_data_train,train_labels_train, cat_features[data_index], num_features[data_index], NB_SPLIT, 2018+(data_index*k))
                #print(proba_test)
                posterior_proba=np.concatenate( [proba_test[j] for j in range(NB_SPLIT)])
                #print(posterior_proba)
                thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC=[],[],[],[],[]
                if(OptimzeParams==True):
                    thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC=Eclair.optimizeThrEntropyRelabelling_nbc(train_data_train, train_labels_train, train_data_val, train_labels_val, posterior_proba, cat_features[data_index], num_features[data_index],nbC)
                    params.append([thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC])
                else :
                    thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC=params_optim[data_index][0],params_optim[data_index][1], params_optim[data_index][2],params_optim[data_index][3],params_optim[data_index][4]
                
                # print([thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC])
                
                    
                perf_u=Data.perfSVClassif_UCI_data(data_uci_Xtrain[data_index], data_uci_ytrain[data_index], 
                                        train_data_train, train_data_val, train_labels_train, train_labels_val, data_uci_Xtest[data_index], data_uci_ytest[data_index], 
                                        cat_features[data_index], num_features[data_index], posterior_proba, 
                                        thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD, param_opt_CM, thr_entr_opt_IC, 2018+(data_index*k), nbC)    
                
                for i_c in range( len(classifiers) ):
                    perf_all[data_index][i_c].append( perf_u[i_c] )

        #perf
        perf_mean_cl=[]
        perf_std_cl=[]
        for m_i in range( len(metrics)):
            perf_mean_cl.append( [] )
            perf_std_cl.append( [] )
            for i_c in range( len(classifiers) ):
                perf_mean_cl[m_i].append( [] )
                perf_std_cl[m_i].append( [] )
                for data_index in range( len(data_uci_names)):
                    perf_mean_cl[m_i][i_c].append( np.mean(np.array(perf_all[data_index][i_c])[:,m_i]) )
                    perf_std_cl[m_i][i_c].append( np.std(np.array(perf_all[data_index][i_c])[:,m_i]) )
        
        
        # print(perf_u)
        # breast_cancer_wd
        # [0.8859649122807017, 0.9342105263157895, 0.9486842105263158, 0.9631578947368422, 0.9824561403508771]
        # [[0.9649122807017544, 0.9649122807017544, 0.9649122807017544, 0.9649122807017544, 0.9649122807017544], [0.9649122807017544, 0.9649122807017544, 0.9649122807017544, 0.9649122807017545, 0.9649122807017544], [0.7894736842105263, 0.8903508771929824, 0.9206140350877194, 0.9508771929824563, 0.9912280701754386], [0.7982456140350878, 0.8947368421052632, 0.9236842105263159, 0.9526315789473685, 0.9912280701754386], [0.7982456140350878, 0.8947368421052632, 0.9236842105263159, 0.9526315789473685, 0.9912280701754386], [0.8859649122807017, 0.9342105263157895, 0.9486842105263158, 0.9631578947368422, 0.9824561403508771]]
        # glass
        # [0.0, 0.16666666666666669, 0.25, 0.3333333333333334, 1.0]
        # [[0.2558139534883721, 0.2558139534883721, 0.2558139534883721, 0.2558139534883721, 0.2558139534883721], [0.2558139534883721, 0.2558139534883721, 0.2558139534883721, 0.25581395348837216, 0.2558139534883721], [0.2558139534883721, 0.2558139534883721, 0.2558139534883721, 0.25581395348837216, 0.2558139534883721], [0.2558139534883721, 0.2558139534883721, 0.2558139534883721, 0.25581395348837216, 0.2558139534883721], [0.2558139534883721, 0.2558139534883721, 0.2558139534883721, 0.25581395348837216, 0.2558139534883721], [0.0, 0.16666666666666669, 0.25, 0.3333333333333334, 1.0]]
        # ionosphere
        # [0.0, 0.5, 0.6499999999999999, 0.7999999999999997, 1.0]
        # [[0.8450704225352113, 0.8450704225352113, 0.8450704225352113, 0.8450704225352113, 0.8450704225352113], [0.7183098591549296, 0.8169014084507042, 0.8464788732394366, 0.8760563380281691, 0.9154929577464789], [0.8591549295774648, 0.8591549295774648, 0.8591549295774648, 0.859154929577465, 0.8591549295774648], [0.8591549295774648, 0.8591549295774648, 0.8591549295774648, 0.859154929577465, 0.8591549295774648], [0.8591549295774648, 0.8591549295774648, 0.8591549295774648, 0.859154929577465, 0.8591549295774648], [0.0, 0.5, 0.6499999999999999, 0.7999999999999997, 1.0]]
        # iris
        # [0.8666666666666667, 0.9222222222222222, 0.941111111111111, 0.9600000000000003, 1.0]
        # [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0000000000000002, 1.0], [1.0, 1.0, 1.0, 1.0000000000000002, 1.0], [1.0, 1.0, 1.0, 1.0000000000000002, 1.0], [1.0, 1.0, 1.0, 1.0000000000000002, 1.0], [0.8666666666666667, 0.9222222222222222, 0.941111111111111, 0.9600000000000003, 1.0]]
        # wine
        # [0.6388888888888888, 0.7499999999999999, 0.7944444444444444, 0.8388888888888891, 0.9722222222222222]
        # [[0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444], [0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444446, 0.9444444444444444], [0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444446, 0.9444444444444444], [0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444446, 0.9444444444444444], [0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444446, 0.9444444444444444], [0.6388888888888888, 0.7499999999999999, 0.7944444444444444, 0.8388888888888891, 0.9722222222222222]]
        # seeds
        # [0.7142857142857143, 0.8055555555555556, 0.8408730158730158, 0.8761904761904764, 0.9761904761904762]
        # [[0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9523809523809523], [0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9523809523809526, 0.9523809523809523], [0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9523809523809526, 0.9523809523809523], [0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9523809523809526, 0.9523809523809523], [0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9523809523809526, 0.9523809523809523], [0.7142857142857143, 0.8055555555555556, 0.8408730158730158, 0.8761904761904764, 0.9761904761904762]]
        # sonar
        # [0.0, 0.25, 0.3625, 0.475, 1.0]
        # [[0.3230769230769231, 0.3230769230769231, 0.3230769230769231, 0.3230769230769231, 0.3230769230769231], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0]]
        # forest
        # [0.0, 0.25, 0.3625, 0.475, 1.0]
        # [[0.3230769230769231, 0.3230769230769231, 0.3230769230769231, 0.3230769230769231, 0.3230769230769231], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0], [0.0, 0.25, 0.3625, 0.475, 1.0]]
        # mice
        # [0.0, 0.125, 0.190625, 0.25625000000000003, 1.0]
        # [[0.6990740740740741, 0.6990740740740741, 0.6990740740740741, 0.6990740740740741, 0.6990740740740741], [0.6759259259259259, 0.6846064814814815, 0.6891637731481481, 0.693721064814815, 0.7453703703703703], [0.6759259259259259, 0.6846064814814815, 0.6891637731481481, 0.693721064814815, 0.7453703703703703], [0.6759259259259259, 0.6846064814814815, 0.6891637731481481, 0.693721064814815, 0.7453703703703703], [0.6759259259259259, 0.6846064814814815, 0.6891637731481481, 0.693721064814815, 0.7453703703703703], [0.0, 0.125, 0.190625, 0.25625000000000003, 1.0]]
        # credit_g
        # [0.23, 0.585, 0.6915000000000002, 0.7980000000000002, 0.94]
        # [[0.62, 0.62, 0.62, 0.62, 0.62], [0.0, 0.5, 0.6500000000000001, 0.8, 1.0], [0.0, 0.5, 0.6500000000000001, 0.8, 1.0], [0.0, 0.5, 0.6500000000000001, 0.8, 1.0], [0.0, 0.5, 0.6500000000000001, 0.8, 1.0], [0.23, 0.585, 0.6915000000000002, 0.7980000000000002, 0.94]]
        # spambase
        # [0.0, 0.5, 0.6500000000000001, 0.8000000000000002, 1.0]
        # [[0.8154180238870793, 0.8154180238870793, 0.8154180238870793, 0.8154180238870793, 0.8154180238870793], [0.8110749185667753, 0.8137893593919653, 0.8146036916395223, 0.8154180238870793, 0.8165038002171553], [0.3485342019543974, 0.5998914223669924, 0.675298588490771, 0.7507057546145495, 0.8512486427795874], [0.3485342019543974, 0.5998914223669924, 0.675298588490771, 0.7507057546145495, 0.8512486427795874], [0.3485342019543974, 0.5998914223669924, 0.675298588490771, 0.7507057546145495, 0.8512486427795874], [0.0, 0.5, 0.6500000000000001, 0.8000000000000002, 1.0]]
        # baseball
        # [0.0, 0.33333333333333326, 0.4666666666666668, 0.6000000000000001, 1.0]
        # [[0.8843283582089553, 0.8843283582089553, 0.8843283582089553, 0.8843283582089553, 0.8843283582089553], [0.8843283582089553, 0.8843283582089553, 0.8843283582089553, 0.8843283582089554, 0.8843283582089553], [0.8843283582089553, 0.8843283582089553, 0.8843283582089553, 0.8843283582089554, 0.8843283582089553], [0.8843283582089553, 0.8843283582089553, 0.8843283582089553, 0.8843283582089554, 0.8843283582089553], [0.8843283582089553, 0.8843283582089553, 0.8843283582089553, 0.8843283582089554, 0.8843283582089553], [0.0, 0.33333333333333326, 0.4666666666666668, 0.6000000000000001, 1.0]]
        
        # Data.barPlotPredictions(barsData=[np.array(perf_mean_cl[j]) *100 for j in range(len(perf_mean_cl))], 
        #                         maxData=[100]*5,
        #                         ytitle='Set-valued Classif', barWidth=0.1, h_lim=[-0.5,6.5], 
        #                         errorData=[np.array(perf_std_cl[j]) *100 for j in range(len(perf_std_cl))],
        #                         classifierNames =['NBC', 'PC', 'ECLAIR', 'SD','CM', '2pic', 'CP', 'NCC', 'NDC'], 
        #                         groupNames=['acc', 'u50', 'u65', 'u80', 'imp'],
        #                         saveFile=dirTMP+data_uci_names[data_index]+'_test_UCI_new_allClassifier.png', y_lim=[20, 101])
        return perf_all, perf_mean_cl, perf_std_cl
    
    def plasticsSortingResults():
        perf_plastics=[]
        #data
        xtrain, xval, ytrain, yval = train_test_split(plasticTrain[plasticTrain.columns[:154]], plasticTrain['class'], test_size=0.15, random_state=10)
        ytrain_num=np.array( np.unique(ytrain, return_inverse=True)[1].tolist() )
        #xy_plasticTrain=pd.concat([xtrain2,ytrain2], axis=1)
        xtest=plasticTest[plasticTest.columns[:154]]
        ytest=plasticTest['class']
        ytest_num=np.array( np.unique(ytest, return_inverse=True)[1].tolist() )
        yval_num=np.array( np.unique(yval, return_inverse=True)[1].tolist() )
        #
        #LDA
        clf = LinearDiscriminantAnalysis()
        clf.fit(xtrain, ytrain)
        xtrain_lda=clf.transform(xtrain)  
        xval_lda=clf.transform(xval)   
        xtest_lda=clf.transform(xtest) 
        nb_f=len(xtrain_lda[0,:])
        #
        start_time = time.time()
        cpts=Util.points_discretization(xtrain_lda, ytrain.tolist(), [t for t in range(nb_f)])
        print("\n --- %s secondes for Discretization ---" % np.round( (time.time() - start_time),3) )
        #
        #df_cpts_plastic = pd.DataFrame(cpts).astype("float")
        #df_cpts_plastic.to_csv(dirTMP+'discretization_plastics.csv',index=False) #save to file
        x_train_disc=Util.discretization(xtrain_lda, cpts,[t for t in range(nb_f)] )
        x_val_disc=Util.discretization(xval_lda, cpts,[t for t in range(nb_f)] )
        x_test_disc=Util.discretization(xtest_lda, cpts,[t for t in range(nb_f)] )
        #
        mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(x_train_disc,ytrain.tolist(), [t for t in range(nb_f)], [], 'gaussian')
        u=np.identity(len(classes))
        preds_bo, preds_eu, preds_proba, preds_proba_sm=nbc.predict(x_test_disc, [t for t in range(nb_f)], [], mu, sigma2, nc, classes, lev, freq, u)
        #sklearn metrics
        nbc.metrics(ytest.tolist(), preds_bo, 'NBC')
        plt.savefig(dirTMP1+'Plastics/acc_NBC_disc_Plastics.png', dpi=300)
        acc_nbc=accuracy_score(ytest.tolist(), preds_bo, normalize=True, sample_weight=None)
        
        perf_plastics.append( [acc_nbc]*5)
        #
        #NCC
        nbC=len(np.unique(ytrain.tolist()))
        N, classes, nc, p_c, lev, freq, p_f=ncc.modelNCC(x_train_disc,  ytrain.tolist())
        ncc_perf_u65, s_opt=ncc.optimize_s(x_val_disc, yval_num, classes, nc, lev, freq)
        #s_opt=0.18367346946938776
        preds=ncc.predict(x_test_disc, s_opt, classes, nc, lev, freq)#s_opt for plastic : 0.18367346946938776
        acc, u50, u65, u80, acc_imp=SVC.setValuedClassEvaluation (ytest_num,preds,len(classes))
        perf_plastics.append( [acc, u50, u65, u80, acc_imp])
        #
        #conformal prediction
        mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(xtrain_lda,  ytrain.tolist(), [], [t for t in range(nb_f)], 'gaussian')
        u=np.identity(len(classes))
        preds_bo_cal, preds_eu_cal, preds_proba_cal, preds_proba_sm_cal=nbc.predict(xval_lda, [], [t for t in range(nb_f)], mu, sigma2, nc, classes, lev, freq, u)
        #preds_bo_val, preds_eu_val, preds_proba_val, preds_proba_sm_val=nbc.predict(train_data_val, [], [t for t in range(154)], mu, sigma2, nc, classes, lev, freq, u)
        preds_bo_test, preds_eu_test, preds_proba_test, preds_proba_sm_test=nbc.predict(xtest_lda, [], [t for t in range(nb_f)], mu, sigma2, nc, classes, lev, freq, u)
        #perf_u65, alpha_optim=SVC.optimizeAlpha_cp_nbc(yval_num, preds_proba_sm_cal, preds_proba_sm_val, train_labels_val, nbC)
        predCP_tst=SVC.conformalPrediction_nbc(yval_num, preds_proba_sm_cal,  preds_proba_sm_test, alpha=0.07)
        acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP=SVC.setValuedClassEvaluation (ytest_num,predCP_tst,nbC)
        print( [acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
        perf_plastics.append( [acc_CONP, u50_CONP, u65_CONP, u80_CONP, acc_imp_CONP])
        #    
        #predictions NDC
        #perf__ndc_u65, beta_optim=ndc.optim_beta_nbc(x_train_disc,ytrain.tolist(), x_val_disc, yval_num, [t for t in range(nb_f)], [], nbC)
        beta_optim=0.0
        posterior_proba=ndc.posteriorTest_nbc(x_train_disc,ytrain.tolist(), [t for t in range(nb_f)], [], x_test_disc)
        pred_ndc=ndc.predict(posterior_proba, beta_optim)
        acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc=SVC.setValuedClassEvaluation (ytest_num,pred_ndc,nbC)
        print( [acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc])
        perf_plastics.append([acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc])
        #
        #predictions PC, ECLAIR and SD
        NB_SPLIT=4
        nbC=4
        y_cal, proba_cal, proba_test=Eclair.trainModelkFolds_nbc(x_train_disc,ytrain_num, [t for t in range(nb_f)], [], NB_SPLIT, 2018)
        posterior_proba=np.concatenate( [proba_test[j] for j in range(NB_SPLIT)])    
        thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD=Eclair.optimizeThrEntropyRelabelling_nbc(x_train_disc,ytrain_num, x_val_disc, yval_num, posterior_proba, [t for t in range(nb_f)], [],nbC)
        #thr_entr_opt_PC, param_opt_ECLAIR, thr_entr_opt_SD=(2.0285714285714285, [2.0285714285714285, 0.0], 0.2)
        ##PC 
        newydata=Eclair.relabellingEntropy (ytrain_num, thr_entr_opt_PC, thr_entr_opt_PC, posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(x_train_disc, newydata, x_test_disc, [t for t in range(nb_f)], [])
        predPC_tst=SVC.pignisticCriterion (masses, np.unique(newydata),nbC)
        acc_PC=accuracy_score(ytest_num, predPC_tst, normalize=True, sample_weight=None)
        print(acc_PC)
        perf_plastics.append( [acc_PC]*5 )
        ##ECLAIR
        newydata=Eclair.relabellingEntropy (ytrain_num, param_opt_ECLAIR[0], param_opt_ECLAIR[0], posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(x_train_disc, newydata, x_test_disc, [t for t in range(nb_f)], [])
        #
        acc_ECLAIR=[0]*50
        u50_ECLAIR=[0]*50
        u65_ECLAIR=[0]*50
        u80_ECLAIR=[0]*50
        acc_imp_ECLAIR=[0]*50
        beta_eclair=np.linspace(1e-10, 0.05, num=50)
        for i in range(len(beta_eclair)):
            print(i)
            predECLAIR=SVC.eclairGFbeta (masses, np.unique(newydata), beta_eclair[i], nbC)#0.005
            acc_ECLAIR[i], u50_ECLAIR[i], u65_ECLAIR[i], u80_ECLAIR[i], acc_imp_ECLAIR[i]=SVC.setValuedClassEvaluation (ytest_num,predECLAIR,nbC)
        #
        #
        plt.plot(beta_eclair, list(zip(acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR )), label=['acc', 'u50', 'u65', 'u80', 'imp'])
        plt.title("Performance d'eclair pour différentes valeurs de $\\beta$")
        plt.xlabel('$\\beta$')
        plt.ylabel('u')
        plt.legend()
        plt.savefig(dirTMP+'beta_eclair.png', dpi=300)
        #
        print( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR])
        perf_plastics.append( [acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR] )
        ##SD
        newydata=Eclair.relabellingEntropy (ytrain_num, thr_entr_opt_SD, thr_entr_opt_SD, posterior_proba, None, nbC, 20)
        masses=Eclair.evidentialPrediction_nbc(x_train_disc, newydata, x_test_disc, [t for t in range(nb_f)], [])
        predSD_tst=SVC.strongDominance (masses, np.unique(newydata),nbC)
        acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (ytest_num,predSD_tst,nbC)
        print( [acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
        perf_plastics.append([acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD])
        #
        cl_names_plas=['NBC', 'NCC', 'Conformal Prediction', 'NDC', 'PiGC', 'ECLAIR', 'StrongD']
        #
        Data.barPlotPredictions(barsData=[np.array(perf_plastics[j]) *100 for j in range(len(perf_plastics))], 
                                maxData=[100]*5,
                                ytitle='Set-valued Classif', barWidth=0.1, h_lim=[-0.5,5], 
                                errorData=[[0] for j in range(len(perf_plastics))],
                                classifierNames =cl_names_plas, 
                                groupNames=['acc', 'u50', 'u65', 'u80', 'imp'],
                                saveFile=dirTMP+'plastic_ALL_test.png', y_lim=[20, 101])
        
        
    def resultsFashionMnist():#results (images) relabelling with conformal prediction
        CP_p_u=Data.svClassifWithConfPredRelabelling_dnn(train_images, train_labels, test_images, test_labels, retrain=False, optimizeAlpha=False, trainCP=False, plotBars=True, dirTMP=dirTMP)  
        ER_p_u=svClassifWithEntropyRelabelling_dnn(xtrain_images, ytrain_images, xtest_images, ytest_images, retrainFalse, optimizeEntr=False, trainCP=False, plotBars=True, dirTMP=dirTMP)

    def resultsUCI():#results (uci) relabelling with conformal prediction
        perf_all, perf_mean_cl, perf_std_cl=Data.svClassifWithEntropyRelabelling_nbc_uci(data_names, data_classes, data_uci_Xtrain, data_uci_Xtest, data_uci_ytrain, data_uci_ytest, cat_features, num_features, OptimzeParams=False)   
        #classifiers=['NBC', 'PC', 'ECLAIR', 'SD','CM', '2pic', 'CP', 'NCC', 'NDC']
        cl_to_plot=[0,8,6, 7, 0, 3, 2]
        cl_names=['NBC', 'NDC', 'CP', 'NCC', 'PigC', 'StrongD', 'ECLAIR']
        #cl_to_plot=[0,1,2,3,6,7,8]
        data_to_plot=[0,2,3,4,5,6,8,9,10,11]
            
        Data.barPlotPredictions(barsData=[ [np.array(perf_mean_cl[2][j])[i] *100 for i in data_to_plot ] for j in cl_to_plot ], 
                                maxData=[100]*len(cl_to_plot),
                                ytitle='u65', barWidth=0.1, h_lim=[-0.5,len(data_to_plot)], 
                                errorData=[ [ np.array(perf_std_cl[2][j][i]) *100 or i in data_to_plot ] for j in cl_to_plot ],
                                classifierNames =cl_names, 
                                groupNames=[data_names[i] for i in data_to_plot],
                                saveFile=dirTMP+'u65_ALL.png', y_lim=[20, 101])    
        
    def predictionsDisplay(file_masses, file_relabel, posteriorProba, test_ydata, beta_ndc, beta_eclair, nbC, dirTMP, title):#this display is for eclair based evidence
        start_time = time.time()
        pred_ndc=Data.ndc(posteriorProba,beta_ndc,nbC)
        acc_ndc, u50_ndc, u65_ndc, u80_ndc, acc_imp_ndc=Data.setValuedClassEvaluation (test_ydata,pred_ndc,nbC)
        print("\n --- %s secondes for NDC ---" % np.round( (time.time() - start_time),3) )
        #predictions and evaluations from csv masses information
        masses_csv2=[]
        with open(file_masses, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                masses_csv2.append( np.array(line,dtype=float) )
        #read new labels        
        masses_newlabels=[]        
        with open(file_relabel, "r") as g:
            reader = csv.reader(g, delimiter=" ")
            for i, line in enumerate(reader):
                masses_newlabels.append( int(line[0]) )
        masses_newlabels.remove(0)
        selflevels=np.unique(masses_newlabels)
        
        start_time = time.time()
        predECLAIR=SVC.eclairGFbeta (masses_csv2[1:], selflevels, beta_eclair, nbC)
        acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=SVC.setValuedClassEvaluation (test_ydata,predECLAIR,nbC)
        print("\n --- %s secondes for Eclair ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predPC_tst=Data.pignisticCriterion (masses_csv2[1:], selflevels,nbC)
        acc_PC=accuracy_score(test_ydata, predPC_tst, normalize=True, sample_weight=None)
        print("\n --- %s secondes for PC ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predSD_tst=SVC.strongDominance (masses_csv2[1:], selflevels,nbC)
        acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=SVC.setValuedClassEvaluation (test_ydata,predSD_tst,nbC)
        print("\n --- %s secondes for SD ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predCM_tst=SVC.convexMixturePrediction (masses_csv2[1:], selflevels,0.7,nbC)
        acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (test_ydata,predCM_tst,nbC)
        print("\n --- %s secondes for CM ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predIC_tst=SVC.intervalCriterion(masses_csv2[1:], selflevels,nbC)
        acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (test_ydata,predIC_tst,nbC)
        print("\n --- %s secondes for IC ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predCP_tst=SVC.conformalPrediction(train_xdata, train_ydata, test_xdata, alpha, ep, nbC)
        acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=SVC.setValuedClassEvaluation (test_ydata,predCP_tst,nbC)
        print("\n --- %s secondes for IC ---" % np.round( (time.time() - start_time),3) )        
        # bars
        barsPC = [acc_PC*100, acc_PC*100, acc_PC*100, acc_PC*100, acc_PC*100]
        barsNDC = [acc_ndc*100, u50_ndc*100, u65_ndc*100, u80_ndc*100, acc_imp_ndc*100]
        barsECLAIR = [acc_ECLAIR*100, u50_ECLAIR*100, u65_ECLAIR*100, u80_ECLAIR*100, acc_imp_ECLAIR*100]
        barsSD = [acc_SD*100, u50_SD*100, u65_SD*100, u80_SD*100, acc_imp_SD*100]
        barsCM = [acc_CM*100, u50_CM*100, u65_CM*100, u80_CM*100, acc_imp_CM*100]
        barsIC = [acc_IC*100, u50_IC*100, u65_IC*100, u80_IC*100, acc_imp_IC*100]
        Data.barPlotPredictions(barsData=[barsPC, barsNDC, barsECLAIR, barsSD, barsCM, barsIC], 
                                maxData=[acc_PC,u50_CM,u65_SD,u80_SD,acc_imp_SD],
                                ytitle='Set-valued Classif', barWidth=0.1, h_lim=[-0.5,6.5], 
                                errorData=[[0]*5, [0]*5, [0]*5, [0]*5, [0]*5, [0]*5],
                               classifierNames =['PC','NDC', 'ECLAIR', 'SD','$CM_{0.7}$', '2pic'], 
                               groupNames=['acc', 'u50', 'u65', 'u80', 'imp'],
                               saveFile=dirTMP+title, y_lim=[65, 101])
    
    
    def predictionsEknnDisplay(F,m, test_ydata, nbC, CM_lambda, dirTMP, title):
        start_time = time.time()
        predPC_tst=Data.pignisticCriterionEknn (m, F,nbC)
        acc_PC=accuracy_score(test_ydata, predPC_tst, normalize=True, sample_weight=None)
        print("\n --- %s secondes for PC ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predSD_tst=Data.strongDominanceEknn (m, F,nbC)
        acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=Data.setValuedClassEvaluation (test_ydata,predSD_tst,nbC)
        print("\n --- %s secondes for SD ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predCM_tst=Data.convexMixturePredictionEknn (m, F,CM_lambda,nbC)
        acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=Data.setValuedClassEvaluation (test_ydata,predCM_tst,nbC)
        print("\n --- %s secondes for CM ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predIC_tst=Data.intervalCriterionEknn(m, F,nbC)
        acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=Data.setValuedClassEvaluation (test_ydata,predIC_tst,nbC)
        print("\n --- %s secondes for IC ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        # bars
        barsPC = [acc_PC*100, acc_PC*100, acc_PC*100, acc_PC*100, acc_PC*100]
        barsSD = [acc_SD*100, u50_SD*100, u65_SD*100, u80_SD*100, acc_imp_SD*100]
        barsCM = [acc_CM*100, u50_CM*100, u65_CM*100, u80_CM*100, acc_imp_CM*100]
        barsIC = [acc_IC*100, u50_IC*100, u65_IC*100, u80_IC*100, acc_imp_IC*100]
        Data.barPlotPredictions(barsData=[barsPC, barsSD, barsCM, barsIC], 
                                maxData=[acc_PC,u50_CM,u65_SD,u80_SD,acc_imp_SD],
                                ytitle='Set-valued Classif', barWidth=0.1, h_lim=[-0.5,6.5], 
                                errorData=[[0]*5, [0]*5, [0]*5, [0]*5],
                               classifierNames =['PC', 'SD','$CM_{0.7}$', '2pic'], 
                               groupNames=['acc', 'u50', 'u65', 'u80', 'imp'],
                               saveFile=dirTMP+title, y_lim=[65, 101])
          
    def barPlotPredictions(barsData, maxData, ytitle, barWidth, h_lim, errorData, classifierNames, groupNames, saveFile, y_lim):
        # width of the bars
        #barWidth = 0.2
        opacity = 0.85
        #yer1 = [0.5, 0.4, 0.5]
        # Choose the height of the error bars (bars2)
        #yer2 = [1, 0.7, 1]
        fig, ax = plt.subplots()
        ax.set(ylim=y_lim)
        name = "tab10"
        #cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        cmap = plt.colormaps.get_cmap(name)
        colors = cmap.colors  # type: list
        ax.set_prop_cycle(color=colors)
        # The x position of bars
        maxData=[]
        #print(barsData[0])
        r1 = np.arange(len(barsData[0]))
        for i in range(len(classifierNames) ):
            # Create blue bars
            #print(colors[i])
            plt.bar(r1, barsData[i], width = barWidth, linewidth=0.25, color = colors[i], 
                    edgecolor = 'black', yerr=errorData[i], 
                    error_kw=dict(lw=0.1, capsize=1, capthick=3), alpha=opacity, 
                    label=classifierNames[i])
            r1 = [x + barWidth for x in r1]
            # Create cyan bars
        #for j in range( len(maxData) ):
            #ax.hlines(y=maxData[j]*100, xmin=j+1, xmax=4.75, linestyles='--', linewidth=1, color='gray')
        ax.hlines(y=np.max(barsData[0]), xmin=h_lim[0], xmax=h_lim[1], linestyles='--', linewidth=1, color='gray')
        #ax.hlines(y=97.5, xmin=h_lim[0], xmax=h_lim[1], linestyles='--', linewidth=1, color='gray')
        #ax.hlines(y=95, xmin=h_lim[0], xmax=h_lim[1], linestyles='--', linewidth=1, color='gray')
        #ax.hlines(y=92.5, xmin=h_lim[0], xmax=h_lim[1], linestyles='--', linewidth=1, color='gray')
        ax.hlines(y=np.max(barsData), xmin=h_lim[0], xmax=h_lim[1], linestyles='--', linewidth=1, color='gray')
        # general layout
        plt.xticks([r + 3*barWidth for r in range(len(barsData[0]))],  groupNames, fontsize='small')
        plt.ylabel(ytitle)
        ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=False, fontsize='xx-small', borderaxespad=3)
        #plt.legend()
        # Show graphic
        #plt.show()
        #plt.savefig(saveFileEPS, format='eps')
        plt.savefig(saveFile, dpi=300)
        
    def barPlotPredictions_R_Results():
        rDIR=r'C:/Users/imoussaten/OwnCloud/Implementations/R/CautiousClassifier/'
        ##### criterion GHC
        flGHC = r"PartialClassification/GHC_25-10-22_Eclair_5M_All.txt";
        ##### criterion GOWAC
        flGOWAC = r"PartialClassification/GOWAC_25-10-22_Eclair_5M_All.txt";
        ##### criterion GRC
        flGRC = r"PartialClassification/GRC_24-10-22_Eclair_5M_All.txt";
        fl2pic = r"outranking/1fold50draws/interval-criterion(last)_5M_All_16_03_2023.txt";
        ##### strong dominance (interval dominance) : EPSILON=0
        flSD = r"idominance/IDOM_eclair_07-10-22_5M_All.txt";
        ##### convex mixture DOMINANCE : EPSILON \in ]0,1]
        #flCM = r"idominance/IDOM_eclair_convexMixture_28-10-22_5M_All.txt";
        flCM = r"idominance/IDOM_eclair_convexMixture_20-03-23_5M_All.txt";#with optimisation based on u50
        ##### weak dominance
        flWeak = r"idominance/WeakDOM_eclair_25-10-22_5M_All.txt";
        ############################################################## start test Fbeta eclair
        flECLAIR = r"eclair/Eclair_Fbeta_27-10-22_5M_All.txt";
        ############################################################## start NDC
        flNDC = r"NDC(NonDeterministicClassifier)/NDC_28-10-22_5M_All.txt";
        ############################################################## start NCC
        flNCC = r"NCC(NaiveCredalClassifier)/NCC_28-10-22_5M_All.txt";
        #flNDC, flNCC
        rfiles=[flSD, flWeak, fl2pic, flCM, flECLAIR, flGHC, flGOWAC, flGRC ]
        #"NDC", "NCC", 
        classifierNames =["Strong dominance","Weak dominance",  "interval-criterion", 
                           "Convex Mixture", "Eclair", "GHC", "GOWAC", "GRC"]
        metrics=['acc', 'u50', 'u65', 'u80', 'imp']
        #original data UCI
        origDataNames = ["Iris", "BC", "Wine", "IS", "DBT", "Glass", "PID", "Sonar", 
                     "Seeds", "Forest", "Ecoli"]
        #Selected data
        selDataNames = ["Wine", "IS", "DBT", "Glass", "Sonar", "Seeds", "Forest"]
        #selDataNames=origDataNames 
        perfUCI=[]
        for i in range( len(classifierNames) ):
            perfUCI.append( Data.readPerfDataUCI(rDIR+rfiles[i]) )
        barsPerfs=[]
        stdPerfs=[]
        for j in range( len(metrics) ):
            barsPerfs_tmp=[]
            stdPerfs_tmp=[]
            for i in range( len(classifierNames) ):
                barsPerfs_tmp_tmp=[]
                stdPerfs_tmp_tmp=[]
                for k in range( len(selDataNames) ):
                    df_describe = pd.DataFrame(perfUCI[i][origDataNames.index(selDataNames[k])])
                    barsPerfs_tmp_tmp.append(  df_describe.describe().iloc[1][j] )
                    stdPerfs_tmp_tmp.append(  df_describe.describe().iloc[2][j] )
                barsPerfs_tmp.append(  barsPerfs_tmp_tmp )
                stdPerfs_tmp.append(  stdPerfs_tmp_tmp )
            barsPerfs.append(  barsPerfs_tmp )
            stdPerfs.append(  stdPerfs_tmp )
        Data.barPlotPredictions(barsData=barsPerfs[1], barWidth=0.1, h_lim=[-0.5,6.5], ytitle='u50',
                                maxData=[50]*11, errorData=stdPerfs[1],
                               classifierNames =classifierNames, groupNames=selDataNames, 
                               saveFile=dirTMP+'u50_eclair_uci_after__a.png', y_lim=[60, 105])
        
    def readPerfDataUCI(fileName):
        perf_CL=[]      
        perf_CL_tmp=[]  
        with open(fileName, "r") as g:
            reader = csv.reader(g, delimiter=" ")
            cc=0
            for i, line in enumerate(reader):
                cc+=1
                perf_CL_tmp.append( np.array(line,dtype=float) )
                if( (cc % 50 ==0) ):
                    np.vstack( perf_CL_tmp )
                    perf_CL_tmp=np.asmatrix( perf_CL_tmp )
                    perf_CL.append( perf_CL_tmp )
                    perf_CL_tmp=[]
        return perf_CL
           
    def augmentationMeanImageClassifier(xtrain_tmp,ytrain_tmp, ep, nbC, intit_label):
        #this part is didicated to create new samples: a new sample from each samples having different classes
        #fix the same number of classes for pairs as for class '0'
        nb_samples_i1_i2=np.sum([k in [0] for k in ytrain_tmp ])
        print( nb_samples_i1_i2 )
        label_pair=intit_label
        for cl_i1 in range( nbC-1 ):#class 1
            print(cl_i1+1)
            i1=cl_i1
            labels_i1=[i for i, x in enumerate(ytrain_tmp) if x == i1]
            for cl_i2 in range(cl_i1+1, nbC ):#class 2
                print(cl_i2+1)
                i2=cl_i2
                labels_i2=[i for i, x in enumerate(ytrain_tmp) if x == i2]
                #compt=0
                #create len(nb_samples_i1_i2) number of classes for {c1,c2}
                pairs_i1_i2=random.sample(list(itertools.product(labels_i1,labels_i2)), nb_samples_i1_i2)
                for i in range( len(pairs_i1_i2) ):
                    ir=pairs_i1_i2[i][0]
                    jr=pairs_i1_i2[i][1]
                    image_mean=[[0.0]*28]*28
                    #the new samples are mean pixels intensity of the two samples
                    for k in range(28):
                        image_mean_tmp=[]
                        for l in range(28):
                            image_mean_tmp.append( (xtrain_tmp[ir][k][l] + xtrain_tmp[jr][k][l])/2 )
                        image_mean[k]=image_mean_tmp
                    xtrain_tmp=np.append(xtrain_tmp,[image_mean], axis=0)
                    ytrain_tmp=np.append(ytrain_tmp,[label_pair],axis=0)
                    # compt=compt+1
                    if( (i%1000==0) ):
                        print('compteur : '+str(i))
                    # if( (compt > nb_samples_i1_i2) ):
                    #     if( (compt > 5000) ):
                    #         print('compteur : '+str(compt))
                    #         print( nb_samples_i1_i2 )
                print( len(xtrain_tmp) )
                print( len(ytrain_tmp) )
                label_pair=label_pair+1
             
        return xtrain_tmp,ytrain_tmp

    def augmentationMaxImageClassifier(xtrain_tmp,ytrain_tmp, ep, nbC, intit_label):
        #this part is didicated to create new samples: a new sample from each samples having different classes
        #fix the same number of classes for pairs as for class '0'
        nb_samples_i1_i2=np.sum([k in [0] for k in ytrain_tmp ])
        print( nb_samples_i1_i2 )
        label_pair=intit_label
        for cl_i1 in range( nbC-1 ):#class 1
            print(cl_i1+1)
            i1=cl_i1
            labels_i1=[i for i, x in enumerate(ytrain_tmp) if x == i1]
            for cl_i2 in range(cl_i1+1, nbC ):#class 2
                print(cl_i2+1)
                i2=cl_i2
                labels_i2=[i for i, x in enumerate(ytrain_tmp) if x == i2]
                #compt=0
                #create len(nb_samples_i1_i2) number of classes for {c1,c2}
                pairs_i1_i2=random.sample(list(itertools.product(labels_i1,labels_i2)), nb_samples_i1_i2)
                for i in range( len(pairs_i1_i2) ):
                    ir=pairs_i1_i2[i][0]
                    jr=pairs_i1_i2[i][1]
                    image_mean=[[0.0]*28]*28
                    #the new samples are mean pixels intensity of the two samples
                    for k in range(28):
                        image_mean_tmp=[]
                        for l in range(28):
                            image_mean_tmp.append( np.max( [xtrain_tmp[ir][k][l],xtrain_tmp[jr][k][l]] ) )
                        image_mean[k]=image_mean_tmp
                    xtrain_tmp=np.append(xtrain_tmp,[image_mean], axis=0)
                    ytrain_tmp=np.append(ytrain_tmp,[label_pair],axis=0)
                    # compt=compt+1
                    if( (i%1000==0) ):
                        print('compteur : '+str(i))
                    # if( (compt > nb_samples_i1_i2) ):
                    #     if( (compt > 5000) ):
                    #         print('compteur : '+str(compt))
                    #         print( nb_samples_i1_i2 )
                print( len(xtrain_tmp) )
                print( len(ytrain_tmp) )
                label_pair=label_pair+1
             
        return xtrain_tmp,ytrain_tmp
    
    def predAugmentations(op_xtrain, op_ytrain, xtest_data, ep, nbC, intit_label):
        #mean_xtrain, mean_ytrain=Data.augmentationMeanImageClassifier(xtrain,ytrain, ep, nbC, intit_label)
        #max_xtrain, max_ytrain=Data.augmentationMaxImageClassifier(xtrain,ytrain, ep, nbC, intit_label)
        #op_xtrain, op_ytrain=Data.augmentationOpImageClassifier(xtrain,ytrain, ep, nbC, intit_label)
        nb_class=len(np.unique(op_ytrain))
        #classification
        BATCH_SIZE = 128
        IMG_D1=28
        IMG_D2=28
        # Model
        model_op = tf.keras.Sequential()
        model_op.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                          activation='relu',
                          kernel_initializer='he_normal',
                          input_shape=(IMG_D1, IMG_D2, 1)))
        model_op.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model_op.add(tf.keras.layers.Dropout(0.25))
        model_op.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model_op.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model_op.add(tf.keras.layers.Dropout(0.25))
        model_op.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model_op.add(tf.keras.layers.Dropout(0.25))
        model_op.add(tf.keras.layers.Flatten())
        model_op.add(tf.keras.layers.Dense(128, activation='relu'))
        model_op.add(tf.keras.layers.Dropout(0.25))
        model_op.add(tf.keras.layers.Dense( nb_class ))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_op.compile(loss=loss_fn,optimizer='adam',metrics=['accuracy'])
        print( model_op.summary() )
        train_model = model_op.fit(max_xtrain, max_ytrain,batch_size=BATCH_SIZE,epochs=20,verbose=1)
        
        probability_model = tf.keras.Sequential([model_op,tf.keras.layers.Softmax()])
        op_postProba = probability_model(xtest_data);
        #op_predModel=[]
        #nbC=10
        setPairs=Data.getSubsetFromPairs(nbC)
        op_levels=[]
        for i in range(nbC):
            op_levels.append( 2**(i) )
        for j in range( len(setPairs) ):
            op_levels.append( Data.b2d([k in setPairs[j] for k in range(nbC) ]) )
            
        # for i in range( len(op_postProba) ):
        #     op_pred=np.argmax( op_postProba[i] )
        #     if( op_pred <10 ):
        #         op_predModel.append( [op_pred] )
        #     else :
        #         op_predModel.append( setPairs[op_pred-nbC]  )
        return op_levels, op_postProba
    
    def predictionsAugmentationDisplay(masses, levels, test_ydata, beta_eclair, nbC, title):
        #predictions and evaluations from csv masses information
        start_time = time.time()
        predECLAIR=Data.eclairGFbeta (masses, levels, beta_eclair, nbC)
        acc_ECLAIR, u50_ECLAIR, u65_ECLAIR, u80_ECLAIR, acc_imp_ECLAIR=Data.setValuedClassEvaluation (test_ydata,predECLAIR,nbC)
        print("\n --- %s secondes for Eclair ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predPC_tst=Data.pignisticCriterion (masses, levels,nbC)
        acc_PC=accuracy_score(test_ydata, predPC_tst, normalize=True, sample_weight=None)
        print("\n --- %s secondes for PC ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predSD_tst=Data.strongDominance (masses, levels,nbC)
        acc_SD, u50_SD, u65_SD, u80_SD, acc_imp_SD=Data.setValuedClassEvaluation (test_ydata,predSD_tst,nbC)
        print("\n --- %s secondes for SD ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predCM_tst=Data.convexMixturePrediction (masses, levels,0.7,nbC)
        acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=Data.setValuedClassEvaluation (test_ydata,predCM_tst,nbC)
        print("\n --- %s secondes for CM ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        predIC_tst=Data.intervalCriterion(masses, levels,nbC)
        acc_IC, u50_IC, u65_IC, u80_IC, acc_imp_IC=Data.setValuedClassEvaluation (test_ydata,predIC_tst,nbC)
        print("\n --- %s secondes for IC ---" % np.round( (time.time() - start_time),3) )
        start_time = time.time()
        # bars
        barsPC = [acc_PC*100, acc_PC*100, acc_PC*100, acc_PC*100, acc_PC*100]
        #barsNDC = [acc_ndc*100, u50_ndc*100, u65_ndc*100, u80_ndc*100, acc_imp_ndc*100]
        barsECLAIR = [acc_ECLAIR*100, u50_ECLAIR*100, u65_ECLAIR*100, u80_ECLAIR*100, acc_imp_ECLAIR*100]
        barsSD = [acc_SD*100, u50_SD*100, u65_SD*100, u80_SD*100, acc_imp_SD*100]
        barsCM = [acc_CM*100, u50_CM*100, u65_CM*100, u80_CM*100, acc_imp_CM*100]
        barsIC = [acc_IC*100, u50_IC*100, u65_IC*100, u80_IC*100, acc_imp_IC*100]
        Data.barPlotPredictions(barsData=[barsPC, barsECLAIR, barsSD, barsCM, barsIC], 
                                maxData=[acc_PC,u50_CM,u65_SD,u80_SD,acc_imp_SD],
                                ytitle='Set-valued Classif', barWidth=0.1, h_lim=[-0.5,6.5], 
                                errorData=[[0]*5, [0]*5, [0]*5, [0]*5, [0]*5],
                               classifierNames =['PC', 'ECLAIR', 'SD','$CM_{0.7}$', '2pic'], 
                               groupNames=['acc', 'u50', 'u65', 'u80', 'imp'],
                               saveFile=dirTMP+title, y_lim=[65, 101])

             
#%% test    
#(train_images, train_labels), (test_images, test_labels)

dirTMP='C:/Users/imoussaten/OwnCloud/Implementations/Python/TensorFlow/eclair/'
#X is the param threshold=0.25 or 0.5
file11='ijar2023/architecture2/rho_0.25_newLabelsFashionMnist2.csv'
file21='ijar2023/architecture2/rho_0.25_PosteriorFashionMnist2.csv'
file31='ijar2023/architecture2/rho_0.25_MassesFashionMnist2.csv'
#choisir le bon dossier




## chercher à comprendre pourquoi certains algorithmes marchent pour certaines données    
###

########### Convex mixture with eclair: building figure lambda evolution on validation data   (fashion images)
NB_S=4
nbC=10
train_images_train, train_images_val, train_labels_train, train_labels_val = train_test_split(train_images, train_labels, test_size=0.15, random_state=201824) 
cal_y, proba_cal, proba_train=Eclair.trainModelkFolds_dnn(train_images_train, train_labels_train, ep=10, nbC=10, NB_SPLIT=NB_S)
posterior_proba=np.concatenate( [proba_train[j] for j in range(NB_S)])

##optimization of lambda
perf_acc_CM=[]
perf_u50_CM=[]
perf_u65_CM=[]
perf_u80_CM=[]
perf_imp_CM=[]
optim_entr_thresh_CM=1.7410714285714286
newydata=Eclair.relabellingEntropy (train_labels_train, optim_entr_thresh_CM, optim_entr_thresh_CM, posterior_proba, None, nbC, 300)
masses=Eclair.evidentialPrediction_dnn (train_images_train, newydata, train_images_val, ep=5)
print("convex Mixture ... ")
#optimize convex Mixture hyper-param
lambda_space=np.linspace(0.0, 1.0, num=100)
for j in range(len(lambda_space)):            
    predCM_tst=SVC.convexMixturePrediction (masses, np.unique(newydata),lambda_space[j],nbC)
    acc_CM, u50_CM, u65_CM, u80_CM, acc_imp_CM=SVC.setValuedClassEvaluation (train_labels_val,predCM_tst,nbC)
    perf_acc_CM.append(acc_CM ), perf_u50_CM.append(u50_CM ), perf_u65_CM.append(u65_CM ), perf_u80_CM.append(u80_CM ), perf_imp_CM.append(acc_imp_CM )

optim_lambda=0.5050505050505051 #(ou 0.7473684210526317)
fig, ax = plt.subplots()
plt.plot(lambda_space, list(zip(perf_acc_CM, perf_u50_CM, perf_u65_CM, perf_u80_CM, perf_imp_CM )), label=['acc', 'u50', 'u65', 'u80', 'imp'])
ax.hlines(y=np.max(perf_u65_CM), xmin=0.0, xmax=1.0, linestyles='--', linewidth=1, color='gray', label='max u65')
plt.title("$CM_\\lambda$ performances")
plt.xlabel('$\\lambda$')
plt.ylabel('u')
plt.legend(loc='upper right')
plt.show()
plt.savefig(dirTMP+'CM/validation_0to1.png', dpi=300)
















#%%
