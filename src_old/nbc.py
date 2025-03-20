# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 08:49:28 2023

@author: imoussaten
"""
#%% 
import pandas as pd
import numpy as np
from scipy.stats import norm#permet de generer une distrib gaussiennne norm(loc = -1., scale = 1.0).pdf([1,25])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
#LabelBinarizer to binarize the target by one-hot-encoding in a OnevsRest fashion
from sklearn.preprocessing import LabelBinarizer
import itertools
#file from the uncertainty quantification project
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


#%% 
class nbc:
    def gaussian(x, mu, sig):
        if(sig>10**-10):
            return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
        else :
            sig=10**-10
            return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    
    def fit_continuos_model(x_train, y_train, cat_feat, num_feat, proba):
        #N=xy_train.shape[0]
        N=len(y_train)
        #col_n=xy_train.columns
        #col_n=len(x_train[0])
        #col_n=len(num_features)+len(cat_features)
        #print('number of num. features '+str(len(num_feat)))
        #classes=xy_train[col_n[-1]].unique()
        classes, counts_cl = np.unique(y_train, return_counts=True)
        #print(classes)
        #print(counts_cl)
        nbclass = len( classes )
        nc = []# number of occurence of each class
        #for ic in range(nbclass):
            #nc.append( np.sum([ xy_train[col_n[-1]].iloc[j]==classes[ic] for j in range(N)]) )
        #numerical features     
        if proba=='gaussian':
            #to acces mu of feature i and classes j: xy_train.groupby(col_n[-1]).mean().loc[classes[j]][i]
            #mu=xy_train.groupby(col_n[-1]).mean()
            mu=[ [ np.mean([ x_train[i,num_feat[j]] for i in range(N) if (y_train[i]==classes[k]) ]) for j in range(len(num_feat))] for k in range(nbclass)]
            #to acces sigma of feature i and classes j: xy_train.groupby( col_n[-1] ).agg(np.std, ddof=0).loc[classes[j]][i]
            #sigma2=xy_train.groupby( col_n[-1] ).agg(np.std, ddof=0) 
            sigma2=[ [ np.std([ x_train[i,num_feat[j]] for i in range(N) if (y_train[i]==classes[k]) ]) for j in range(len(num_feat))] for k in range(nbclass)]
        #categorical features 
        lev=[]
        freq=[]
        for ic in range(nbclass):
            lev_ic=[]
            freq_ic=[]
            for cat_f in range(len(cat_feat)):
                lev_tmp, counts_tmp=np.unique([x_train[k,cat_feat[cat_f]]  for k in range(N) if (y_train[k]==classes[ic]) ], return_counts=True)
                lev_ic.append( lev_tmp )
                freq_ic.append( [counts_tmp[t]/np.sum(counts_tmp) for t in range(len(counts_tmp))] )
            lev.append( lev_ic )
            freq.append( freq_ic )
        for ic in range(nbclass):
            nc.append(counts_cl[ic]/N)
        return mu, sigma2, nc, classes, lev, freq
    
    def softmax(v):#produit des nan quand les valeurs sont trop grandes
        proba=[]
        for i in range(len(v)):
            proba.append( np.exp(v[i])/(np.sum(np.exp(v))) )
        return proba
    
    def predict(x_test, cat_feat, num_feat, mu, sigma2, nc, classes, lev, freq, utilities):
        #mu, sigma2, nc, classes, col_n=nbc.fit_continuos_model(xy_data, probabilityModel)
        #print(mu)
        #print(sigma2)
        nbclass=len(classes)
        nbtest=len(x_test)
        preds_proba_sm = np.zeros( (nbtest, nbclass) )
        preds_proba = np.zeros( (nbtest, nbclass) )
        preds_bo = []
        preds_eu = []
        #nba = col_n
        nba=len(num_feat)+len(cat_feat)
        for ir in range(nbtest):
            z=[]
            for ia in range(nba):
                z.append( x_test[ir][ia] )
            pic = []
            max_pic=0
            for ic in range(nbclass):
                pic_tmp = nc[ic]
                #print('first '+str(pic_tmp))
                for i_nf in range(len(num_feat)):
                    pic_tmp *=nbc.gaussian(z[num_feat[i_nf]], mu[ic][i_nf], sigma2[ic][i_nf])
                    #print('x= '+str(z[ia])+', mu= '+str(mu[ic][ia])+' and sigma= '+str(sigma2[ic][ia]))
                    #print(pic_tmp)
                for i_cf in range(len(cat_feat)):
                    #print(lev[ic][i_cf])
                    #print(z[cat_features[i_cf]])
                    
                    idx = np.where(np.array(lev[ic][i_cf]) == z[cat_feat[i_cf]])
                    if( len(idx[0])==0 ):
                        #print('value '+str(z[cat_feat[i_cf]])+' is not encountred in learning data !')
                        pic_tmp *=10**-10
                    else :
                        pic_tmp *=freq[ic][i_cf][idx[0][0]]
                    
                if max_pic < pic_tmp:
                    max_pic = pic_tmp  
                pic.append(pic_tmp)
            preds_bo.append( classes[np.argmax(pic)] )
            preds_proba[ir]= pic
            #softmax produit des nan quand les valeurs sont trop grandes
            preds_proba_sm[ir]=nbc.softmax([pic[j]-max_pic for j in range(nbclass)])
            eu=[]
            for ic in range(nbclass):
                eu.append(  np.sum([ utilities[t][ic]*preds_proba[ir][t] for t in  range(nbclass) ])  )
            preds_eu.append( classes[np.argmax( eu )] )
        return preds_bo, preds_eu, preds_proba, preds_proba_sm
    
    def metrics(actual, predicted, h):#h is the name of classifier
        classes=np.unique(actual)
        print(accuracy_score(predicted, actual, normalize=True, sample_weight=None))
        # confusion matrix
        cnf_matrix = confusion_matrix(actual, predicted)
        print('Matrice de confusion : \n',cnf_matrix)
        #classes = range(0,len(actual.unique()))
        
        plt.figure()
        plt.imshow(cnf_matrix, interpolation='nearest',cmap='Oranges')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")
        plt.ylabel('Actual labels')
        plt.xlabel('Predicted Labels with '+h)
        plt.show()
        # outcome values order in sklearn (binary case)
        if len(np.unique(actual))==2:
            tn, fp, fn ,tp = confusion_matrix(actual, predicted).reshape(-1)
            print('Values of TN, FP, FN, TP : \n',tn, fp, fn, tp)
        # classification report for precision, recall f1-score and accuracy
        matrix = classification_report(actual, predicted)
        print('Classification  Report: \n',matrix)
    
    def rocMC(y_train, y_test, proba, orderClasses, class_of_interest):
        #roc curve and precision-recall curves
        #ROC curves are appropriate when observations are balanced between each class, 
        #while precision-recall curves are appropriate for unbalanced datasets
        #False positive (fp). Predicting an event when it didn't happen.
        #False negative (fn). Do not predict an event when there has been an event.
        #Area under the curve ROC (AUC)
        #Smaller values on the x-axis of the graph indicate lower false positives and 
        #higher true negatives. Larger values on the y-axis indicate higher true 
        #positives and lower false negatives.
        #A “skillful” model will assign a higher probability to an actual positive 
        #occurrence chosen at random than to a negative occurrence on average.
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        y_onehot_test.shape  # (n_samples, n_classes)
        label_binarizer.transform([class_of_interest])
        print(label_binarizer.classes_)
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
        class_id
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            proba[:, orderClasses][:, class_id],#faire attention a l ordre des classes dans la prediction classes=ytrain.unique()
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
        )
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC curves:\n"+class_of_interest+" vs others")
        plt.legend()
        plt.show()
        
    def roc_binary(y_test, y_proba, class_of_interest, algoML):
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y_test))]
        # keep probabilities for the positive outcome only
        lr_probs = y_proba[:, 1]
        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('No skills : ROC AUC=%.3f' % (ns_auc))
        print('Continuous NBC : ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        #class_of_interest=y_test.unique()[0]
        bytest1=pd.Series(np.where(y_test.values == class_of_interest, 1, 0),y_test.index)
        ns_fpr, ns_tpr, _ = roc_curve(bytest1, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(bytest1, lr_probs)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No skills')
        plt.plot(lr_fpr, lr_tpr, marker='.', label=algoML)
        # axis labels
        plt.xlabel('FP rate')
        plt.ylabel('TP rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

        

        
#%% 

# #tests
# #Import iris data
# dirTMP='C:/Users/imoussaten/OwnCloud/Implementations/Python/TensorFlow/eclair/'
# irisFile='Plastics/iris.csv'
# iris=pd.read_csv(dirTMP+irisFile, encoding = "ISO-8859-1", sep = ';', decimal=",", low_memory=False)
# #cp=nbc.points_discretization(iris)
# #disc_iris=nbc.discretization(iris,cp)

# xtrain, xtest, ytrain, ytest = train_test_split(iris[iris.columns[:4]], iris['Species'], test_size=0.2, random_state=10)
# xy_irisTrain=pd.concat([xtrain,ytrain], axis=1)
# mu, sigma2, nc, classes, col_n=nbc.fit_continuos_model(xy_irisTrain, 'gaussian')
# u3=[[1,0,0],
#     [0,1,0],
#     [0,0,1]
#     ]
# preds_bo, preds_eu, preds_proba=nbc.predict(xy_irisTrain, xtest, False,mu, sigma2, nc, classes, col_n, u3)

# #sklearn metrics
# #accuracy=len([i for i, j in zip(preds_class, ytest) if i == j])/len(ytest)
# nbc.metrics(ytest, preds_bo, ['c','a','b'])
# #general roc for multiclass problem
# #multi class (iris)
# nbc.rocMC(ytrain, ytest, preds_proba, [2,0,1], 'a')



# #Import spam data
# spamFile='Plastics/spam.csv'
# spam=pd.read_csv(dirTMP+spamFile, encoding = "ISO-8859-1", sep = ';', decimal=",", low_memory=False)

# xtrain1, xtest1, ytrain1, ytest1 = train_test_split(spam[spam.columns[:57]], spam['type'], test_size=0.2, random_state=10)
# xy_spamTrain=pd.concat([xtrain1,ytrain1], axis=1)
# mu, sigma2, nc, classes, col_n=nbc.fit_continuos_model(xy_spamTrain, 'gaussian')
# u1=[[0.95,0],[0.76,1]]
# u2=[[0.95,0],[0.57,1]]
# preds_bo1, preds_eu1, preds_proba1=nbc.predict(xy_spamTrain, xtest1, False,mu, sigma2, nc, classes, col_n, u2)

# #what is missed with bo as nonspam: many are 50% s 50% ns
# probaNonSpam1=[ preds_proba1[t][0] for t in range(len(ytest1)) if ((ytest1.iloc[t]=='nonspam') & (preds_bo1[t]=='spam'))]
# #what is missed with eu as nonspam
# probaNonSpam2=[ preds_proba1[t][0] for t in range(len(ytest1)) if ((ytest1.iloc[t]=='nonspam') & (preds_eu1[t]=='spam'))]

# plt.plot(range(len(probaNonSpam1)), probaNonSpam1, marker="o")
# #utilities 0 1 0.95 (\pi_1=0.95) and 0.95*0.8 (\pi_2=0.8)
# #   ns  s
# #ns 0.95 0
# #s  0.95*0.6 1

# tn=len([ t for t in range(len(ytest1)) if ((ytest1.iloc[t]=='spam') & (preds_bo1[t]=='spam')) ])
# fp=len([ t for t in range(len(ytest1)) if ((ytest1.iloc[t]=='spam') & (preds_bo1[t]=='nonspam')) ])
# fn=len([ t for t in range(len(ytest1)) if ((ytest1.iloc[t]=='nonspam') & (preds_bo1[t]=='spam')) ])
# tp=len([ t for t in range(len(ytest1)) if ((ytest1.iloc[t]=='nonspam') & (preds_bo1[t]=='nonspam')) ])
# tn,fp,fn,tp
# #metrics
# #len([i for i, j in zip(preds_eu1, ytest1) if i == j])/len(ytest1)
# nbc.metrics(ytest1, preds_bo1, ['nonspam','spam'])
# nbc.metrics(ytest1, preds_eu1, ['nonspam','spam'])
# #roc curve for binary class (spam)
# nbc.roc_binary(ytest1, preds_proba1, 'spam', 'Continuous NBC')


# #apply gaussian Naive bayes of sklearn
# gnb = GaussianNB()
# y_pred = gnb.fit(xtrain1,ytrain1).predict( xtest1)
# #metrics
# nbc.metrics(ytest1, y_pred)

# #apply  Multinomial Naive bayes of sklearn
# clf = MultinomialNB(force_alpha=True)
# clf.fit(xtrain1,ytrain1) 
# pred_MN=clf.predict(xtest1)
# #metrics
# nbc.metrics(ytest1, pred_MN)


# #Import plastic data
# dirTMP1='C:/Users/imoussaten/OwnCloud/Implementations/Python/TensorFlow/eclair/'
# plasticFile='Plastics/PlasticsTrain.csv'
# plastics=pd.read_csv(dirTMP1+plasticFile, encoding = "ISO-8859-1", sep = ';', decimal=",", low_memory=False)

# wavenumber=[]
# for i in range(154):
#     wavenumber.append( int(plastics.columns[i][:-4]) )
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(wavenumber, plastics.iloc[100][:154])


# xtrain2, xtest2, ytrain2, ytest2 = train_test_split(plastics[plastics.columns[:154]], plastics['class'], test_size=0.2, random_state=10)
# #xy_plasticTrain=pd.concat([xtrain2,ytrain2], axis=1)

# mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(xtrain2.values,ytrain2.tolist(), [], [t for t in range(154)], 'gaussian')
# u=np.identity(len(classes))
# preds_bo, preds_eu, preds_proba, preds_proba_sm=nbc.predict(xtest2.values, [], [t for t in range(154)], mu, sigma2, nc, classes, lev, freq, u)

#  np.sum( (ytest2.tolist()[j]=='PE') for j in range(len(ytest2)) )           
# #preds_bo2, preds_eu2, preds_proba2=nbc.predict(xy_plasticTrain, xtest2, False,mu, sigma2, nc, classes, col_n, u4)
# #sklearn metrics
# nbc.metrics(ytest2.tolist(), preds_bo, 'NBC')
# plt.savefig(dirTMP1+'Plastics/acc_NBC_Plastics.png', dpi=300)
# #general roc for multiclass problem
# #multi class (iris)
# #nbc.rocMC(ytrain2.tolist(), ytest2.tolist(), preds_proba, [2,1,3,0], 'PE')


# plt.plot(range(len(preds_proba2)), preds_proba2[:len(preds_proba2),0], marker="o")


# gnb = GaussianNB()
# pred = gnb.fit(xtrain2,ytrain2).predict( xtest2)
# nbc.metrics(ytest2.tolist(), pred,classes, 'NBC')

# proba_pred = gnb.fit(xtrain2,ytrain2).predict_proba( xtest2)

# plt.plot(range(len(proba_pred)), proba_pred[:len(proba_pred),0], marker="o")

# from sklearn import svm
# clf = svm.SVC(decision_function_shape='ovo',probability=True)
# clf.fit(xtrain2,ytrain2)
# predsvm=clf.predict(xtest2)
# nbc.metrics(ytest2.tolist(), predsvm, 'SVM')
# plt.savefig(dirTMP1+'Plastics/acc_SVM_Plastics.png', dpi=300)

# pred2_svm=clf.predict_proba(xtest2) #### Attention les proba n'ont pas de sens pour svm
# plt.plot(range(len(pred2_svm)), pred2_svm[:len(pred2_svm),0], marker="o")


