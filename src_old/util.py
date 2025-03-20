# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:32:36 2023

@author: imoussaten
"""

# this class contains all math methods that are useful for the uncertainty quantificaion project
#natural representation of sets theory: b2d, d2b, b2set, getSubsetFromPairs
#distances:
#graph theory: kernelGraph,
#Belief functions:

#######################  A. Imoussaten

#%% imports
import numpy as np
import pandas as pd
from mdlp import mdlp

#%% util class
class Util:
    def b2d (b):
        dec = 0
        tmp = 0
        for i in range(1,len(b)+1):
            tmp=int( b[i-1] * 2**(i-1) )
            dec = int(dec)
            dec += tmp
        return dec
    def d2b (dec,l):
        b = [False]*l
        dec_tmp = int(dec)
        for i in reversed(range(1, l+1)):
            b[i-1] = (dec_tmp // 2**(i-1) ) ==1;# // integer division
            dec_tmp = int(dec_tmp)
            dec_tmp = dec_tmp % 2**(i-1)
        return b
    def b2set (b,l):
        cl = []
        for i in range(l):
            if( b[i] ):
                cl.append( i )
        return cl
    def getSubsetFromPairs(nbC):#ndices of pairs start from nbC
        Set_pair=[]
        for cl_i1 in range( nbC-1 ):
            for cl_i2 in range(cl_i1+1, nbC ):#class 2
                Set_pair.append( [cl_i1, cl_i2] )
        return Set_pair
        
    #point prediction classifiers
    # L2 square distance between two vectorized images x and y
    def distance1(x,y):
        return np.sum(np.square(x-y))
    #L2 distance between two vectorized images x and y
    def distance2(x,y):
        return np.sqrt(np.sum(np.square(x-y)))
    #and can be coded as below
    def distance3(x,y):
        return np.linalg.norm(x-y)
    
    ######### graph theory
    def kernelGraph(g_tmp, nbC):
        N=[]#kernel
        tags=[]
        listPredec=[]#non-empty predecessors
        for k in range(nbC):
            listPredec.append( g_tmp.predecessors(k) )
            
        emptyPred=[False]*nbC    
        while (emptyPred!=[True]*nbC):
            for k in range(nbC):
                if( emptyPred[k]==False ):
                    if( (len(g_tmp.predecessors(k))==0) ): 
                        N.append( g_tmp.vs['name'][k] )
                        emptyPred[k]=True
                        for j in range(len(g_tmp.successors( k ))):
                           tags.append(g_tmp.successors( k )[j] )
            for ktag in range( len(tags)):
                emptyPred[tags[ktag]]=True
                for kpred in range(nbC):
                    if( emptyPred[k]==False ):
                        if( [tt in listPredec[kpred] for tt in [tags[ktag]]]==[True] ):
                            listPredec[kpred].remove( tags[ktag] )
            return N
        
    ######### Belief functions
    def prob2be (p, selflevels):#get body of evidence (F,M) from masses p
        #epsilon=math.ulp(1.0);
        F=[]#focal elements
        m=[]#masses associated to focal elements
        for i in range (len(p)):
            if( p[i]>10**(-4) ):
                F.append(selflevels[i])
                m.append( p[i] )
        m=m/np.sum(m)
        m.tolist()
        return F,m
    def getBelPl(F,m, nbC):#getpl, bel, pig from body of evidence
        pl=[0.0]*nbC
        bel=[0.0]*nbC
        pig=[0.0]*nbC
        for i in range (len(F)):
            bF_tmp=Util.d2b(F[i],nbC)#binary vector of F[i]
            for j in range (nbC):
                bC_tmp=[k in [j+1] for k in range(1,nbC+1)]#binary vector of class j
                if( bF_tmp[j]==bC_tmp[j] ):# j in F[i]
                    pl[j]+=m[i]
                    pig[j]+=m[i]/(np.sum(bF_tmp))
                    if( np.sum(bF_tmp)==1 ):# [j] equal to F[i]
                        bel[j]=m[i]
        return bel,pl, pig 
    ##########################################
    #DISCRETIZATION ##########################
    def points_discretization(x_train_data, y_train_data, num_feat):
        cut_points_mdlp = []
        #num_vars = xy_train_data.shape[1]-1
        num_vars = len(x_train_data[0,:])
        #col_n=xy_train_data.columns
        #target=col_n[num_vars]
        # discretization with MDLP.
        for i in range(num_vars):
            if( i in num_feat):
                #cut_points_mdlp.append( mdlp.cut_points( xy_train_data[col_n[i]], xy_train_data[target]) )
                cut_points_mdlp.append( mdlp.cut_points( x_train_data[:,i], y_train_data) )
            else :
                cut_points_mdlp.append(np.array([]))   
        return cut_points_mdlp
    
    def discretization(x_data, cut_points_mdlp, num_feat):
        #cut points
        #cut_points_mdlp=Util.points_discretization(x_data, y_data)
        #col_n=xy_data.columns
        # apply point discretization to data
        #disc_data = pd.DataFrame(columns=col_n[:len(col_n)-1])
        disc_data = []
        num_att = len(x_data[0,:])
        num_ex = len(x_data) 
        #print((num_att,num_ex))
        #for ir in range(xy_data.shape[0]):#row
        for ir in range( num_ex ):#row
            #disc_col_ic=[]
            disc_data.append( [] )
            #for ic in range( (xy_data.shape[1])-1 ):#col
            for ia in range( num_att ):#col
                #if( (ir==13) & (num_ex==36) ):
                    #print( ia )
                if( ia in num_feat):
                    nb_p_ia=len(cut_points_mdlp[ia])
                    # if( (ir==13) & (num_ex==36) ):
                    #     print(cut_points_mdlp[ia])
                    #     print(x_data[ir][ia])
                    if ( nb_p_ia==0 ):
                        disc_data[ir].append( 0 )
                        # if( (ir==13) & (num_ex==36) ):
                        #     print('yes')
                    else :
                        # if( (ir==13) & (num_ex==36) ):
                        #     print('yes nb_p_ia = '+str(nb_p_ia))
                        for idp in range(nb_p_ia):
                            #print(str(ir)+' '+str(ic)+' '+str(idp))
                            #print( xy_data[col_n[ir]].iloc[ic] )
                            #print ( cut_points_mdlp[ic][idp] )
                            if x_data[ir][ia] <= cut_points_mdlp[ia][idp]:
                                disc_data[ir].append( idp )
                                break
                        # if( (ir==13) & (num_ex==36) ):
                        #     print('yes and idp ='+str(idp))
                        if x_data[ir][ia] > cut_points_mdlp[ia][nb_p_ia-1]:#-1 for the last value
                            disc_data[ir].append( nb_p_ia )
                            # if( (ir==13) & (num_ex==36) ):
                            #     print('yes')
                    # if( (ir==13) & (num_ex==36) ):
                    #     print( 'add : '+str(disc_data[ir][ia]) )
                else :
                    disc_data[ir].append( x_data[ir][ia] )
                    
                            
            #print( len(disc_col_ic) )
            #disc_data.loc[len(col_n)-1] = disc_col_ic    
            #print( disc_col_ic )
            # if( len(disc_data[ir])!=num_att ):
            #     print('problem at ir ='+str(ir))
            #     print(x_data[ir])
            #     print(disc_data[ir])
        #disc_data[col_n[-1]]=xy_data[col_n[-1]]
        #print(disc_data)
        return np.array(disc_data)    
    
#%% test
