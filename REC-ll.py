# REC version containing results of ll
import pandas as pd
import numpy as np
import math
import sklearn
from numpy import array
from numpy import linalg as LA

# a function to find the normalized clustering indicator
def NCI(K, pred, nK): 
    #K: number of classes 
    #nK: a vector indicating # of obs in each class (length of K)
    #pred: a prediction vector starting from 1
    n=len(pred)
    P=np.zeros((n,K))
    
    for i in range(n):
        ind=pred[i]
        P[i,ind-1]=1/math.sqrt(nK[ind-1])
    
    return P

def NCImax(M): 
    #M: a matrix
    n=M.shape[0]
    K=M.shape[1]
    P=np.zeros((n,K))
    
    for i in range(n):
        temp=np.argmax(M[i,:], axis=0)
        P[i,temp]=1
    
    return P

# a function to scale back P
def scaleP(K, P, nK):
    #K: number of classes 
    #nK: a vector indicating # of obs in each class (length of K)
    #P: unscaled P
    N=P.shape[0]
    ans=np.zeros((N,K))
    
    for i in range(N):
        temp=np.argmax(P[i,:], axis=0)
        ans[i,temp]=1/math.sqrt(nK[temp])
        
    return ans

# a function to return prediction vector
def labelP1(P):
    #P: unscaled P
    N=P.shape[0]
    ans=np.zeros(N)
    
    for k in range(N):
        m = np.argmax(P[k,:], axis=0)
        ans[k]=m+1
        
    return ans

# a function that finds nK by a matrix
def find_nK(P,K):
    row_max=np.argmax(P, axis=1)
    ind_sum=np.ones(K)
    for j in range(K):
        ind_sum[j]=(row_max==j).sum()
        
    return ind_sum

# a function that finds nK by a vector
def find_nK2(p,K):
    #ind_sum=np.ones(K)
    #for j in range(K):
    #    ind_sum[j]=(p==j+1).sum()
        
    unique, counts = np.unique(p, return_counts=True)
        
    return counts    

# a function to calculate piecewise multiplication of two 3D arrays
def pieceM(data1,data2):
    M=data1.shape[0]
    N=data1.shape[1]
    K=data1.shape[2]
    
    res=np.empty((M,N,K))
    for i in range(M):
        res[i,:,:]=data1[i,:,:]@data2[i,:,:]
        
    return np.sum(res,axis = 0)

# a function to compare accuracy by the chisquare distance
def accuracy(P1,P2):
    F=LA.norm(np.matmul(P1,np.transpose(P1))-np.matmul(P2,np.transpose(P2)))
    ans=F**2
    
    return ans

# a function that calculates sum_i(P0-P^1W^1-...-P^{m-1}W^{m-1}-P^{m+1}W^{m+1}-...)*P^m
def except_m(M,data,W,j,P0):
    W_new=np.delete(W,M,axis=0)
    data_new=np.delete(data,M,axis=0)
    
    tem=pieceM(data_new,W_new)
    temp=P0[:,j]-tem[:,j]
    #temp=np.matmul((P0-tem),data[M,:,:])
    
    part1_sum=np.dot(temp,data[M,:,j])
    #part1_sum=sum(temp[:,j])
    return part1_sum

# A function to calculate gradient for PSG
def PSG_g(m,data,W,j,P0):
    tem=pieceM(data,W)
    temp=tem[:,j]-P0[:,j]
    
    part1_sum=np.dot(temp,data[m,:,j])
    return part1_sum

# a function that calculates sum_i(P^1W^1+...+P^{m-1}W^{m-1}+P^{m+1}W^{m+1}+...)*P^m
def except_m2(M,data,W,j):
    W_new=np.delete(W,M,axis=0)
    data_new=np.delete(data,M,axis=0)
    
    tem=pieceM(data_new,W_new)
    part1_sum=np.dot(tem[:,j],data[M,:,j])
    return part1_sum

# a function takes M summations of except_m divided by sum^K(P_ij^2)
def sum_m(M,data,W,j,P0):
    res=np.ones(M)
    for i in range(M):
        res[i]=except_m(i,data,W,j,P0)/np.sum(data[i,:,j]**2)
    
    result=sum(res)
    return result

# for ADMM
def sum_m2(M,data,W,j,P0):
    res=np.ones(M)
    for i in range(M):
        res[i]=except_m(i,data,W,j,P0)
    
    result=sum(res)
    return result

def for_c(data,j,M):
    temp=np.ones(M)
    for m in range(M):
        temp[m]=1/np.sum((data[m,:,j])**2)
    
    return np.sum(temp)

def findW(M,K,data,P0,W): #for one iteration only
    #M: the number of replications/algorithms/singles
    #data: a 3-D array containing P^i: i=1,...,M
    What=np.zeros((M,K,K))
    
    for m in range(M):
        for j in range(K):
            c=for_c(data,j,M)
            part_1=except_m(m,data,W,j,P0)
            What[m,j,j]=(part_1-(sum_m(M,data,W,j,P0)-1)/c)/np.sum(data[m,:,j]**2)
            
    return What

# for ADMM
def findW2(M,K,data,P0,W,Z,U,rho,lmda): #for one iteration only
    #What=np.zeros((M,K,K))
    What=np.ones(K*M)

    for m in range(M):
        for j in range(K):
            ind=m*K+j
            num=rho*(Z[m,j,j]-U[m,j,j])-lmda-2*sum_m2(m,data,W,j,P0)
            den=2*np.sum(data[m,:,j]**2)+rho
            #What[m,j,j]=num/den
            What[ind]=num/den
            
    return What

# multiple iterations until convergence: training W
def full(nrep,M,K,data,P0):
    #para below randomly initialize W (ND array)
    #rd=np.random.dirichlet(np.ones(M),size=1).ravel()
    #a=np.zeros((K,K))
    #W=np.zeros((M,K,K))
    #for i in range(M):
    #    np.fill_diagonal(a, rd[i])
    #    W[i,:,:]=a
    w=np.diag(np.array(np.repeat(1/M, K)))
    W=np.repeat(w[None,...],M,axis=0)
    
    counter=0
    ll=np.zeros((nrep))
    while counter<nrep:
        W_hat=findW(M,K,data,P0,W)
        ll[counter]=accuracy(pieceM(data,W_hat),P0)
        counter=counter+1
        W=W_hat
        
    #W[W < 0] = 0   
    return W,ll

# a function to update W by ADMM
def ADMM_REC(nrep,M,K,data,P0,rho,lmda):
    rd=np.random.dirichlet(np.ones(M),size=1).ravel()
    a=np.zeros((K,K))
    W=np.zeros((M,K,K))
    for i in range(M):
        np.fill_diagonal(a, rd[i])
        W[i,:,:]=a
    Z=W/2
    U=W/rho #store values & initialization
    
    counter=1
    while counter<=nrep:
        Wh=findW2(M,K,data,P0,W,Z,U,rho,lmda) 
        
        What=np.zeros((M,K,K))
        for i in range(M):
            np.fill_diagonal(What[i,:,:],Wh[K*i:(K*i+K-1)])
            #W[i,:,:]=What[K*i:(K*i+K-1)]
            
        #print(What)
        Zhat=What+U
        Zhat[Zhat<0]=0
        Uhat=U+What-Zhat
        W=What
        Z=Zhat
        U=Uhat
        
        counter=counter+1
    
    return W

# projected sub gradient method for lasso
def gradient(W,data,lmda,m,j,P0):
    g=2*PSG_g(m,data,W,j,P0)+lmda*np.sign(W[m,j,j])
    return g           

def PSG(W_int,M,K,data,P0,alpha,lmda):
    #alpha is the step size, lmda is to tune L1 norm
    W=W_int
    #a=np.diag(np.array(np.repeat(1/M, K)))
    #a=np.repeat(a[None,...],M,axis=0)
    nrep=1
    if sum(sum(sum(W_int<0)))>0:
        nrep=800
        a=np.diag(np.array(np.repeat(1/M, K)))
        W=np.repeat(a[None,...],M,axis=0)

    counter=0
    while counter<nrep:# or error>th:
        What=np.zeros((M,K,K))
        for m in range(M):
            for j in range(K):
                g=gradient(W,data,lmda,m,j,P0)
                temp=max(W[m,j,j]-alpha*g,0)
                #temp=W[m,j,j]-alpha*g
                What[m,j,j]=temp
        counter=counter+1
        
        W=What
        
    return W

# New: a function to find the normalized clustering indicator
def NCI2(P): 
    P=P/P.sum(axis=1)[:,None] #divide the rowsums
    col=np.sqrt(P.sum(axis=0))
    P=np.divide(P,col)
    
    return P

# projected sub gradient method for nonnegative
def gradient2(W,data,m,j,P0):
    g=2*PSG_g(m,data,W,j,P0)#+lmda*np.sign(W[m,j,j])
    return g           

def PSG2(W_int,M,K,data,P0,alpha):
    #alpha is the step size
    W=W_int
    nrep=1
    if sum(sum(sum(W_int<0)))>0:
        nrep=800
        a=np.diag(np.array(np.repeat(1/M, K)))
        W=np.repeat(a[None,...],M,axis=0)

    counter=0
    ll=np.zeros((nrep))
    while counter<nrep:# or error>th:
        What=np.zeros((M,K,K))
        for m in range(M):
            for j in range(K):
                g=gradient2(W,data,m,j,P0)
                temp=max(W[m,j,j]-alpha*g,0)
                #temp=W[m,j,j]-alpha*g
                What[m,j,j]=temp
        ll[counter]=accuracy(pieceM(data,What),P0)
        counter=counter+1
        
        W=What
        
    return W,ll
            
        
    
    
    