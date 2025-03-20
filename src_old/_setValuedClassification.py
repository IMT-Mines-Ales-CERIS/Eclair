# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:14:39 2023

@author: imoussaten
"""

# this code is part of uncertainty quantification project
# this contains the code for set-valued classification (SVC) methods
#######################  A. Imoussaten

#%% imports
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
##graph theory imports
#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import connected_components
import igraph as ig
#file from the uncertainty quantification project
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from util import Util
from nbc import nbc
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian
import itertools


#%% SetValuedClassification class
class SVC:
    def setValuedClassEvaluation (truth,pred,nbC): 
        #truth contains the indices of examples class
        #pred contains the subsets (subset of elements from 0 to nbC-1)
        acc=0.0
        u50=0.0
        u65=0.0
        u80=0.0
        acc_imp=0.0
        if( len(truth)!=len(pred) ):
            print('truth and pred must have th same length')
        else:
             inPred=[0.0]*len(truth)
             z=[0.0]*len(truth)#discounted Acc
             z65=[0.0]*len(truth)
             z80=[0.0]*len(truth)
             z_acc=[0.0]*len(truth)
             for i in range ( len(truth) ):
                 #if( len(pred[i])==0 ):#if empty set is predicted it's taken for ignorance (conformal prediction)
                     #pred[i]=np.arange(nbC).tolist()
                 if( (len(pred[i])>0) & ([k in pred[i] for k in [truth[i]]]==[True]) ):
                     inPred[i]=1.0
                     z[i]=1.0/len(pred[i])
                     z65[i]=-0.6*(z[i]**2)+1.6*z[i]
                     z80[i]=-1.2*(z[i]**2)+2.2*z[i]
                     if( len(pred[i])==1 ):
                         z_acc[i]=1.0
             acc=np.sum(z_acc)/len(truth)
             u50=np.sum(z)/len(truth)
             u65=np.sum(z65)/len(truth)
             u80=np.sum(z80)/len(truth)
             acc_imp=np.sum(inPred)/len(truth)
             
        return acc, u50, u65, u80, acc_imp
    
    def pignisticCriterion (m_test, selflevels,nbC):#eclair
        pred=[]
        for i in range (len(m_test)):
            F,m=Util.prob2be(m_test[i], selflevels)
            bel,pl, pig=Util.getBelPl(F,m,nbC)
            max_pig=np.max(pig)
            #pred.append( np.argmax(pig) )
            pred.append( [j for j in range(len(pig)) if(pig[j]>=max_pig) ] )
        return pred     
    
    def pignisticCriterionEknn (m, F,nbC):
        pred=[]
        for i in range (len(m)):
            bel,pl, pig=Util.getBelPl(F[i],m[i],nbC)
            pred.append(  np.argmax(pig) )
        return pred  


    def weightOWATolEntrpy(gamma, k):
        # Define the evaluation function
        # def eval_f(x):
        #     obj = np.sum(x * np.log(x))
        #     grad = np.log(x) + 1.0
        #     print(obj)
        #     print(grad)
        #     return obj, grad
        def fun(x):
            if( np.sum( [x[i]>0 for i in range(len(x)) ])==len(x)):
                return (- np.sum(x * np.log(x)), - np.log(x) - 1.0)
            else:
                #print('this should not happen !')
                return (1e+10,1.0)

        # Define the equality constraint function
        # def eval_g_eq(x):
        #     bn_cst=2
        #     constr = np.array([ np.sum([( ( (k-(i+1)) / (k - 1) ) * x[i]) - gamma for i in range(0, k)]), np.sum(x)-1 ])
        #     jacobian = np.zeros((bn_cst,k))
        #     jacobian[1, :] = 1.0
        #     jacobian[0, :] = [ (k-(i+1)) / (k - 1) for i in range(0, k)]
        #     print(constr)
        #     print(jacobian)
        #     print(constr.shape)
        #     print(jacobian.shape)
        #     return constr, jacobian
        def linear_constraint(x):
            return np.array([ np.sum([( ( (k-(i+1)) / (k - 1) ) * x[i]) - gamma for i in range(k)]), np.sum(x)-1 ])
        def jacobian(x):
            bn_cst=2
            jac = np.zeros((bn_cst,k))
            jac[1, :] = 1.0
            jac[0, :] = [ (k-(i+1)) / (k - 1) for i in range(0, k)]
            return jac
        def fun_der(x):
            return Jacobian(lambda x: fun(x)[0])(x).ravel()
        
        def fun_hess(x):
            return Hessian(lambda x: fun(x)[0])(x)
            
            
        if k == 1:
            return np.array([1.0])
        elif k == 2:
            return np.array([gamma, 1 - gamma])
        else:
            # Initial values
            x0 = np.ones(k) / k
    
            # Bounds
            bounds = [(0, 1.0) for _ in range(k)]
    
            #Minimize the objective function subject to the equality constraint
            result = minimize(fun, x0, method='trust-constr', jac=True,#hess=lambda x: hess_f(x), 
                               constraints={'type': 'eq', 'fun': linear_constraint}, bounds=bounds)
            #result = minimize(lambda x: fun(x)[0], x0, method='dogleg', jac=fun_der, hess=fun_hess, constraints={'type': 'eq', 'fun': linear_constraint}, bounds=bounds)
            #result = minimize(obj_f, x0, method='trust-constr', bounds=bounds, hessp=hess_f, jac=jacobian, constraints=lambda x: linear_constraint(x))
            
            return result.x
        
    def owa(x, w):#owa operator
        x_sorted = sorted(x, reverse=True)
        return sum(xi * wi for xi, wi in zip(x_sorted, w))
    
    
    def UtilityMatrixOWAExtension(u, weights):#extending a utility matrix to subsets
        n = len(u)
        spowset = 2**n - 1
        uext = np.ones((spowset, n))
        for i in range(1, spowset + 1):
            #print(i)
            K = Util.b2set(Util.d2b(i, n),n)
            #print(K)
            for j in range(n):
                #print(j)
                x = u[K, j]
                uext[i-1, j] = SVC.owa(x, weights[len(K)-1])
                #print(x)
                #print(weights[len(K)-1])
                #print(uext[i-1, j])
        return uext
    
    def generalized_hurwicz_criterion(u_ext, m, alpha):
        #partial classification via complete pre-orders among partial assignments
        # Generalized Hurwicz criterion (ma and Denoeux, 2021)
        # This criterion considers a convex
        # combination of the minimum and maximum utility,
        # with a pessimism index \alpha \in [0, 1] adjusting the combination.
        # The generalized maximin and maximax criteria
        # are special cases corresponding, respectively, to \alpha=1 and \alpha=0.
        #
        #u is the initial utility matrix
        #gamma the owa parameter
        #m the mass function
        #K and G are two subsets to compare (given in natural order: na)
        #return 1 if K dominates G and 0 elsewhere
        n = u_ext.shape[1]
        spowset = 2**n - 1
        max_e = 0.0
        k_opt = 0
        for k in range(spowset):
            e_k = 0.0
            for i in range(spowset):
                b = Util.b2set(Util.d2b(i+1, n),n)
                e_k += m[i] * (alpha * np.min(u_ext[k, b]) + (1 - alpha) * np.max(u_ext[k, b]))
            print(e_k)
            if e_k > max_e:
                max_e = e_k
                k_opt = k
    
        return k_opt
    
    
    def generalized_owa_criterion(u_ext, m, weights):
        # Generalized OWA criterion (ma and Denoeux, 2021)
        n = u_ext.shape[1]
        spowset = 2**n - 1
        max_e = 0.0
        k_opt = 0
    
        for k in range(spowset):
            e_k = 0.0
            for i in range(spowset):
                b = Util.b2set(Util.d2b(i+1, n),n)
                #print(b)
                #print(m[i])
                #print(u_ext[k, b])
                #print(weights[len(b)-1])
                e_k += m[i] * SVC.owa(u_ext[k, b], weights[len(b)-1])
    
            if e_k > max_e:
                max_e = e_k
                k_opt = k
    
        return k_opt
    
    def generalized_regret_criterion(u_ext, m):
        #This criterion extends
        #Savages minimax regret criterion
        
        #u is the initial utility matrix
        #gamma the owa parameters for extending u
        #m the mass function
        #K and G are two subsets to compare (given in natural order: na)
        #return 1 if K dominates G and 0 elsewhere
        n = u_ext.shape[1]
        spowset = 2**n - 1
        r = np.zeros((spowset, n))
    
        for i in range(1, spowset + 1):
            for j in range(n):
                r[i-1, j] = np.max(u_ext[:, j]) - u_ext[i-1, j]
    
        min_e = float('inf')
        k_opt = 0
    
        for k in range(spowset):
            r_k = 0.0
            for i in range(1, spowset + 1):
                b = Util.b2set(Util.d2b(i, n),n)
                r_k += m[i-1] * np.max(r[k, b])
    
            if r_k < min_e:
                min_e = r_k
                k_opt = k
    
        return k_opt

    def predection_genCriteria(masses, unique_new_labels, u, criterion, gamma_all, alpha_fhc, beta_gowac):#criterion=[True,True,True] (ghc, gowac, grc)
        n = u.shape[1]
        #ps = 2**n - 1  # power set
        ghc_pred = []
        gowac_pred = []
        grc_pred = []
    
        weights_gamma = [SVC.weightOWATolEntrpy(gamma_all, k) for k in range(1, n + 1)]
        if criterion[1] == True:
            weights_beta = [SVC.weightOWATolEntrpy(beta_gowac, k) for k in range(1, n + 1)]
        else:
            weights_beta = None
    
        u_ext = SVC.UtilityMatrixOWAExtension(u, weights_gamma)
    
        for t in range(len(masses)):
            mt = masses[t, :]
            if criterion[0] == True:
                ghc_pred.append( SVC.generalized_hurwicz_criterion(u_ext, mt, alpha_fhc) )
            elif criterion[1] == True:
                gowac_pred.append( SVC.generalized_owa_criterion(u_ext, mt, weights_beta) )
            elif criterion[2] == True:
                grc_pred.append( SVC.generalized_regret_criterion(u_ext, mt) )
            else:
                print("Criterion not specified !!!!")
    
        return ghc_pred, gowac_pred, grc_pred
    
    
    
    #concerning parameters' optimization, use  param_grid and library GridSearchCV
    #from sklearn.model_selection import GridSearchCV
    # param_grid = {"max_depth":    [4, 5, 6],
    #               "n_estimators": [200, 300, 400],
    #               "learning_rate": [0.01, 0.015]}
    # try out every combination of the above values
    # search = GridSearchCV(regressor, param_grid, cv=5).fit(xtrain, ytrain)
    # regressor=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
    #                            n_estimators  = search.best_params_["n_estimators"],
    #                            max_depth     = search.best_params_["max_depth"],
    #                            eval_metric='rmsle')
    


    def strongDominance (m_test, selflevels, nbC):#eclair
        pred=[]
        for i in range (len(m_test)):
            N=[False]*nbC
            F,m=Util.prob2be(m_test[i], selflevels)
            bel,pl, pig=Util.getBelPl(F,m,nbC)
            for a in range (nbC):
                comp=0
                for b in range (nbC):
                    if(a!=b):
                        if(  (bel[b] >= pl[a]) ):
                            break
                        else:
                            comp+=1
                if( (comp==(nbC-1)) ):
                    N[a]=True
            #print(N)
            pred.append( Util.b2set (N,nbC) ) #Data.b2d(N) )
        return pred
    
    def weakDominance (m_test, selflevels, nbC):#eclair
        pred=[]
        for i in range (len(m_test)):
            N=[False]*nbC
            F,m=Util.prob2be(m_test[i], selflevels)
            bel,pl, pig=Util.getBelPl(F,m,nbC)
            for a in range (nbC):
                comp=0
                for b in range (nbC):
                    if(a!=b):
                        #is b weak dominates a 
                        if( (bel[b] >= bel[a]) & (pl[b] >= pl[a]) ):
                            if( (bel[b] > bel[a])  | (pl[b] > pl[a]) ):
                                break
                        else:
                            comp+=1
                if( (comp==(nbC-1)) ):
                    N[a]=True
            #print(N)
            pred.append( Util.b2set (N,nbC) ) #Data.b2d(N) )
        return pred
    
    def strongDominanceEknn (m, F, nbC):
        pred=[]
        for i in range (len(m)):
            N=[False]*nbC
            bel,pl, pig=Util.getBelPl(F[i],m[i],nbC)
            for a in range (nbC):
                comp=0
                for b in range (nbC):
                    if(a!=b):
                        if(  (bel[b] >= pl[a]) ):
                            break
                        else:
                            comp+=1
                if( (comp==(nbC-1)) ):
                    N[a]=True
            #print(N)
            pred.append( Util.b2set (N,nbC) ) #Data.b2d(N) )
        return pred
    
    def intervalCriterion(m_test, selflevels,nbC):#eclair
        pred=[]
        sclass =[]
        for k in range(nbC):
            sclass.append( ' '+str(k)+' ' )  
        for i in range (len(m_test)):
            N=[]
            N_tmp=[]
            S=[]#outranking relation
            F,m=Util.prob2be(m_test[i], selflevels)
            bel,pl, pig=Util.getBelPl(F,m,nbC)
            for a in range (nbC-1):
                for b in range (a+1,nbC):
                    if( ( (pig[a] <= pl[b]) & (pig[b] <= pl[a]) ) | (pig[a]>pl[b]) ):
                        S.append([a,b])
                    if( ( (pig[a] <= pl[b]) & (pig[b] <= pl[a]) ) | (pig[b]>pl[a]) ):
                        S.append([b,a])                        
            
            #print(S)
            gS=ig.Graph (S,directed=True)
            gS.vs['name']=sclass
            Adj=[]
            Adj=gS.get_adjacency()
            components = gS.components(mode='strong')
            if( (len( set(components.membership) )== nbC) ):
                #single kernel
                N_tmp=Util.kernelGraph(gS, nbC)
                #print(N)
            else:#multiple kernels
                gS.contract_vertices(components.membership, combine_attrs='concat')
                gS.simplify(multiple=True, loops=True, combine_edges='concat')
                N_tmp=Util.kernelGraph(gS, len(gS.vs['name']))

            #print(N_tmp)
            N_str=''.join(N_tmp)
            N_str=N_str.split()
            #print(N_str)
            for iN in range (len(N_str)):
                i_N_str=int(N_str[iN])
                c_N_str=0
                for iNAdj in range (len(N_str)):
                    if( Adj[i_N_str,int(N_str[iNAdj])]==1 ):
                        c_N_str+=1
                if( (c_N_str==len(N_str)-1) ):# if a P b, b is removed from N
                    N.append(  i_N_str )
            #print(N)    
            pred.append( N ) #Data.b2d(N) )            
        return pred
    
    def intervalCriterionEknn (m, F,nbC):
        pred=[]
        sclass =[]
        for k in range(nbC):
            sclass.append( ' '+str(k)+' ' )  
        for i in range (len(m)):
            N=[]
            N_tmp=[]
            S=[]#outranking relation
            bel,pl, pig=Util.getBelPl(F[i],m[i],nbC)
            for a in range (nbC-1):
                for b in range (a+1,nbC):
                    if( ( (pig[a] <= pl[b]) & (pig[b] <= pl[a]) ) | (pig[a]>pl[b]) ):
                        S.append([a,b])
                    if( ( (pig[a] <= pl[b]) & (pig[b] <= pl[a]) ) | (pig[b]>pl[a]) ):
                        S.append([b,a])                        
            
            #print(S)
            gS=ig.Graph (S,directed=True)
            gS.vs['name']=sclass
            Adj=[]
            Adj=gS.get_adjacency()
            components = gS.components(mode='strong')
            if( (len( set(components.membership) )== nbC) ):
                #single kernel
                N_tmp=Util.kernelGraph(gS, nbC)
                #print(N)
            else:#multiple kernels
                gS.contract_vertices(components.membership, combine_attrs='concat')
                gS.simplify(multiple=True, loops=True, combine_edges='concat')
                N_tmp=Util.kernelGraph(gS, len(gS.vs['name']))

            #print(N_tmp)
            N_str=''.join(N_tmp)
            N_str=N_str.split()
            #print(N_str)
            for iN in range (len(N_str)):
                i_N_str=int(N_str[iN])
                c_N_str=0
                for iNAdj in range (len(N_str)):
                    if( Adj[i_N_str,int(N_str[iNAdj])]==1 ):
                        c_N_str+=1
                if( (c_N_str==len(N_str)-1) ):# if a P b, b is removed from N
                    N.append(  i_N_str )
            #print(N)    
            pred.append( N ) #Data.b2d(N) )            
        return pred
    
    def convexMixturePrediction (m_test, selflevels, epsilon, nbC):#eclair
        pred=[]
        for i in range (len(m_test)):
            N=[False]*nbC
            F,m=Util.prob2be(m_test[i], selflevels)
            bel,pl, pig=Util.getBelPl(F,m,nbC)
            for a in range (nbC):
                comp=0
                cma=epsilon*pig[a] + (1-epsilon)*pl[a]
                for b in range (nbC):
                    if(a!=b):
                        cmb=epsilon*pig[b] + (1-epsilon)*bel[b]
                        if(  (cmb >= cma) ):
                            break
                        else:
                            comp+=1
                if( (comp==(nbC-1)) ):
                    N[a]=True
            #print(N)
            pred.append( Util.b2set (N,nbC) ) #Data.b2d(N) )
        return pred
    
    def optim_convexMixturePredictionEclair(test_ydata, nbC,file_relabel, file_masses,saveFile):
        #get eclair information
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
        
        
        epsilonSpace=[k/50 for k in range(51)]
        perf_CM=[ [0.0 for i in range(len(epsilonSpace)) ] for j in range(5)]
        for i in range (len(epsilonSpace)):
            #print('epsilon is '+str(epsilonSpace[i]))
            predCM_tst=SVC.convexMixturePrediction (masses_csv2[1:], selflevels,epsilonSpace[i],nbC)
            perf_CM[0][i], perf_CM[1][i], perf_CM[2][i], perf_CM[3][i], perf_CM[4][i]=SVC.setValuedClassEvaluation (test_ydata,predCM_tst,nbC)
            #print('peerformance is '+str(perf_CM[i]))
        #plot

        fig, ax = plt.subplots()
        #ax.set(ylim=y_lim)
        name = "tab10"
        #cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        cmap = plt.colormaps.get_cmap(name)
        colors = cmap.colors  # type: list
        ax.set_prop_cycle(color=colors)
        ax.plot(epsilonSpace, perf_CM[0], color=colors[0], label='acc')
        ax.plot(epsilonSpace, perf_CM[1], color=colors[1], label='u50')
        ax.plot(epsilonSpace, perf_CM[2], color=colors[2], label='u65')
        ax.plot(epsilonSpace, perf_CM[3], color=colors[3], label='u80')
        ax.plot(epsilonSpace, perf_CM[4], color=colors[4], label='imp')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True, fontsize='xx-small')
        plt.savefig(saveFile, dpi=300)
        return epsilonSpace[ np.argmax(perf_CM[3]) ], np.max(perf_CM[3])
    
    def convexMixturePredictionEknn (m, F, epsilon, nbC):#isBody is boolean to check if m is represented as a body of evidence
        pred=[]
        for i in range (len(m)):
            N=[False]*nbC
            bel,pl, pig=Util.getBelPl(F[i],m[i],nbC)
            for a in range (nbC):
                comp=0
                cma=epsilon*pig[a] + (1-epsilon)*pl[a]
                for b in range (nbC):
                    if(a!=b):
                        cmb=epsilon*pig[b] + (1-epsilon)*bel[b]
                        if(  (cmb >= cma) ):
                            break
                        else:
                            comp+=1
                if( (comp==(nbC-1)) ):
                    N[a]=True
            #print(N)
            pred.append( Util.b2set (N,nbC) ) #Data.b2d(N) )
        return pred
    
    def optim_convexMixturePredictionEknn(test_ydata, m, F, nbC, num):#,saveFile):
        lambda_space=np.linspace(0.0, 1.0, num)
        perf_CM=[ [0.0 for i in range(len(lambda_space)) ] for j in range(5)]
        for i in range (len(lambda_space)):
            if(i%10==0):
                print('epsilon is '+str(lambda_space[i]))
            predCM_tst=SVC.convexMixturePredictionEknn (m, F, lambda_space[i], nbC)
            perf_CM[0][i], perf_CM[1][i], perf_CM[2][i], perf_CM[3][i], perf_CM[4][i]=SVC.setValuedClassEvaluation (test_ydata,predCM_tst,nbC)
            #print('peerformance is '+str(perf_CM[i]))
        #plot

        # fig, ax = plt.subplots()
        # #ax.set(ylim=y_lim)
        # name = "tab10"
        # #cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        # cmap = plt.colormaps.get_cmap(name)
        # colors = cmap.colors  # type: list
        # ax.set_prop_cycle(color=colors)
        # ax.plot(epsilonSpace, perf_CM[0], color=colors[0], label='acc')
        # ax.plot(epsilonSpace, perf_CM[1], color=colors[1], label='u50')
        # ax.plot(epsilonSpace, perf_CM[2], color=colors[2], label='u65')
        # ax.plot(epsilonSpace, perf_CM[3], color=colors[3], label='u80')
        # ax.plot(epsilonSpace, perf_CM[4], color=colors[4], label='imp')
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        #   ncol=2, fancybox=True, shadow=True, fontsize='xx-small')
        # plt.ylabel('$lambda$')
        # #plt.savefig(saveFile, dpi=300)
        return lambda_space, perf_CM
    
    def eclairGFbeta (m_test, selflevels, beta, nbC):
        pred=[]
        for i in range (len(m_test)):
            #print('m_test = '+str(m_test[i]))
            F,m=Util.prob2be(m_test[i], selflevels)
            #print('F = '+str(F))
            #print('m = '+str(m))
            bel,pl, pig=Util.getBelPl(F,m,nbC)
            gain=[]
            for j in range (len(F)):
                gain_tmp=0.0
                for k in range (len(F)):
                    A=Util.b2set(Util.d2b(F[j],nbC), nbC)
                    B=Util.b2set(Util.d2b(F[k],nbC), nbC) 
                    inters=list(set( A )  & set( B ) )
                    Fbeta=( (1+beta**2)*len(inters)) / ( ((beta**2)*len(B)) + len(A) )
                    gain_tmp=gain_tmp+Fbeta*m[k]
                gain.append( gain_tmp )
            pred.append(   Util.b2set(Util.d2b(F[np.argmax(gain)],nbC), nbC) )
        return pred 
    
    def conformalPrediction_dnn_old(train_xdata, train_ydata, test_xdata, alpha, ep, nbC):#new version exist for all kind of data
        ##### presentin conformal prediction
        # l+m examples
        #m is the number of training ex, k the number of calibration ex
        # conformal prediction algo:
        # # 1- we divide training data to training X1 (l) and calibration X2 (k)
        # # 2- a classifier/regressor delta is learnt using X1
        # # 3- non-conformity on calibration (strangeness) measure is computed on X2 by 
        # # # comparing the prediction of delta and the true value : : s_i=1-p(.|x_i)[y_i]
        # # 4- non-conformity alpha_y for new example x is computed as in 3-, 
        # # where the true class is each potential class y (all classes are tested) 
        # # 5- calculate quantile of level \hat{q}=[(n+1)(1-alpha)]/n of scores s_i
        # # 6- for a given alpha>0 (level 1-alpha), (epsilon is 1% or 5%)
        # # # the predictive region output is : {y: s_x < \hat{q} }
        #####
        # 1- 
        #We split the train set in train and calibration set. 
        # The calibration set will be 20% from the original train set, 
        # therefore the split will be train/calibration of 0.8/0.2.
        RANDOM_STATE = 2018
        TEST_SIZE = 0.2
        X_train, X_cal, y_train, y_cal = train_test_split(train_xdata, train_ydata, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        m_train=len(y_train) #0.8*len(train_ydata)
        k_cal=len(y_cal) #0.2*len(train_ydata)
        
        ###
        # 2- a classifier/regressor ==> DNN
        # parameters regarding DNN
        BATCH_SIZE = 128
        IMG_D1=28
        IMG_D2=28
        #Train the model
        ##Build the model
        ### We will use a Sequential model.
        # Model
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
        model_cpc.add(tf.keras.layers.Dense( nbC ))
        #
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_cpc.compile(loss=loss_fn,optimizer='adam',metrics=['accuracy'])
        #inspect the model
        print( model_cpc.summary() )
        #
        #Run the model
        #validation_data=??
        model_cpc.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=ep, verbose=1)
        probability_model_cpc = tf.keras.Sequential([model_cpc,tf.keras.layers.Softmax()])
        cpc_proba_cal = probability_model_cpc(X_cal);
        cpc_proba_test = probability_model_cpc(test_xdata);
        
        ###
        # 3- non-conformity on calibration (strangeness) measure is computed on X2 by 
        # # # comparing the prediction of delta and the true value : : s_i=1-p(.|x_i)[y_i]
        # # here for calibration examples we use the true class of examples
        print("Start measuring non-conformity for calibration")
        n=len(y_cal)
        cal_scores=[]
        for i in range(n):
            cal_scores.append( 1-cpc_proba_cal[i][y_cal[i]] )
        ## 4- the non-conorfmity score for its potential class is calculated as in 3 but
        ## prediction for test examples
        # 2: get adjusted quantile
        print("Start get adjusted quantile")
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(cal_scores, q_level, interpolation='higher')
        print( 'qhat= '+str(qhat))
        # #code for quantile : empiracal quantile
        # #cumulative function
        # F_cal_scores=[ len([i for i in range(len(cal_scores)) if (cal_scores[i]<=cal_scores[j] )])/len(cal_scores) for j in range(len(cal_scores))]
        # #quantile function
        # Q_q_level=np.min([ cal_scores[j] for j in range(len(cal_scores)) if (F_cal_scores[j]>=q_level)]) 
        
        #prediction for test_xdata
        print("Start predictions")
        prediction_sets = cpc_proba_test >= (1-qhat) # 3: form prediction sets
        cp_pred=[]
        for i in range(test_xdata.shape[0]):
            #if( len([j for j, x in enumerate( prediction_sets[i] ) if x])==0 ):
                #print( str(qhat) + ' :: ' +str(cpc_proba_test[i]) )
            cp_pred.append( [j for j, x in enumerate( prediction_sets[i] ) if x] )
        #    
        #    
        #tf.keras.backend.clear_session()
        #del model_cpc
        print(" ... end !")
        return cp_pred 
    
    def conformalPrediction(y_cal, proba_cal, proba_test, alpha):
        ##### presentin conformal prediction
        # l+m examples
        #m is the number of training ex, k the number of calibration ex
        # conformal prediction algo:
        # # 1- we divide training data to training X1 (l) and calibration X2 (k)
        # # 2- a classifier/regressor delta is learnt using X1
        # # 3- non-conformity on calibration (strangeness) measure is computed on X2 by 
        # # # comparing the prediction of delta and the true value : : s_i=1-p(.|x_i)[y_i]
        # # 4- non-conformity alpha_y for new example x is computed as in 3-, 
        # # where the true class is each potential class y (all classes are tested) 
        # # 5- calculate quantile of level \hat{q}=[(n+1)(1-alpha)]/n of scores s_i
        # # 6- for a given alpha>0 (level 1-alpha), (epsilon is 1% or 5%)
        # # # the predictive region output is : {y: s_x < \hat{q} }
        #####
        # 1- 
        #We split the train set in train and calibration set. 
        # The calibration set will be 20% from the original train set, 
        # therefore the split will be train/calibration of 0.8/0.2.
        # 2- a classifier/regressor ==> nbc or DNN
        ###
        # 3- non-conformity on calibration (strangeness) measure is computed on X2 by 
        # # # comparing the prediction of delta and the true value : : s_i=1-p(.|x_i)[y_i]
        # # here for calibration examples we use the true class of examples
        #print("Start measuring non-conformity for calibration")
        n=len(y_cal)
        cal_scores=[]
        for i in range(n):
            cal_scores.append( 1-proba_cal[i][y_cal[i]] )
        ## 4- the non-conorfmity score for its potential class is calculated as in 3 but
        ## prediction for test examples
        # 2: get adjusted quantile
        #print("Start get adjusted quantile")
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(cal_scores, q_level, method='higher')
        #print( 'qhat= '+str(qhat))
        # #code for quantile : empiracal quantile
        # #cumulative function
        # F_cal_scores=[ len([i for i in range(len(cal_scores)) if (cal_scores[i]<=cal_scores[j] )])/len(cal_scores) for j in range(len(cal_scores))]
        # #quantile function
        # Q_q_level=np.min([ cal_scores[j] for j in range(len(cal_scores)) if (F_cal_scores[j]>=q_level)]) 
        
        #prediction for test_xdata
        #print("Start predictions")
        prediction_sets = proba_test >= (1-qhat) # 3: form prediction sets
        cp_pred=[]
        for i in range( len(prediction_sets) ):
            #if( len([j for j, x in enumerate( prediction_sets[i] ) if x])==0 ):
                #print( str(qhat) + ' :: ' +str(proba_test[i]) )
            cp_pred.append( [j for j, x in enumerate( prediction_sets[i] ) if x] )
        #    
        #    
        #tf.keras.backend.clear_session()
        #del model_cpc
        #print(" ... end !")
        return cp_pred    
    
    def optimizeAlpha_cp(y_cal, preds_proba_sm_cal, preds_proba_sm_test, train_labels_val, nbC):
        alpha_space=np.linspace(0.05, 0.075, num=10)
        cp_perf_u65=[]
        for i in range(len(alpha_space)):
            pred_cp=SVC.conformalPrediction(y_cal, preds_proba_sm_cal, preds_proba_sm_test, alpha_space[i])
            acc, u50, u65, u80, acc_imp=SVC.setValuedClassEvaluation (train_labels_val,pred_cp,nbC)
            cp_perf_u65.append( u65 )
        i_opt=np.argmax(cp_perf_u65)
        return cp_perf_u65, alpha_space[i_opt]
            
            
#%% test

# train_data_train, train_data_val, train_labels_train, train_labels_val = train_test_split(data_uci_Xtrain[0], data_uci_ytrain[0], test_size=0.15, random_state=201824)
# nbC=len(np.unique(data_uci_ytrain[0]))

# X_train, X_cal, y_train, y_cal = train_test_split(train_data_train, train_labels_train, test_size=0.2, random_state=2018)
# mu, sigma2, nc, classes, lev, freq=nbc.fit_continuos_model(X_train, y_train, cat_features[0], num_features[0], 'gaussian')
# u=np.identity(len(classes))
# preds_bo_cal, preds_eu_cal, preds_proba_cal, preds_proba_sm_cal=nbc.predict(X_cal, cat_features[0], num_features[0], mu, sigma2, nc, classes, lev, freq, u)
# preds_bo_val, preds_eu_val, preds_proba_val, preds_proba_sm_val=nbc.predict(train_data_val, cat_features[0], num_features[0], mu, sigma2, nc, classes, lev, freq, u)
# preds_bo_test, preds_eu_test, preds_proba_test, preds_proba_sm_test=nbc.predict(data_uci_Xtest[0], cat_features[0], num_features[0], mu, sigma2, nc, classes, lev, freq, u)
        
# perf_u65, alpha_optim=SVC.optimizeAlpha_cp_nbc(y_cal, preds_proba_sm_cal, preds_proba_sm_val, train_labels_val, nbC)

# pred_cp=SVC.conformalPrediction_nbc(y_cal, preds_proba_sm_cal, preds_proba_sm_test, alpha_optim)

# acc, u50, u65, u80, acc_imp=SVC.setValuedClassEvaluation (data_uci_ytest[0],pred_cp,nbC)


