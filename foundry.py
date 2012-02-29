#!/usr/bin/python
'''Module containing the objective function and constraints for the FB capacity loading optimization'''
''''''
import sys
import numpy
import re


class foundry:

    def __init__(self):
	'''Number of days'''
	self.I=31
	'''Number of FB'''
	self.J=3
	'''Numbers of FM in each of the FBs'''
	self.K=numpy.array([12,9,7],dtype=int)
	self.Kmax=self.K.max()
	'''Number of orders and kinds of production'''
	self.L=36
	''' Rigidity coeefficient'''
	self.alpha=0.5
	'''Productivities for change'''
	self.V=numpy.array([40,40,40,45,45,45,50,45,40,40,\
			    41,41,40,45,45,47,50,45,40,50,\
                            49,48,40,43,41,45,49,45,40,49,\
                            40,40,50,45,45,44],dtype=int)
        ''' Vilumes of orders'''
        self.W=numpy.array([20,2501,30,450,45,145,450,4500,4000,400,\
			    2241,12241,940,745,445,2047,1231,145,540,550,\
                            5449,148,140,3543,6441,5545,2249,345,440,549,\
                            140,140,150,145,451,1201],dtype=int)
        '''Ability of the FB's to produce kinds of production'''
        self.z=numpy.array([[1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1], \
    			    [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0], \
    			    [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 1,1,1,0,0,0]  \
    			    ],dtype=int)
	self.notz=1-self.z
	''' Terms of finishing production'''
	self.T=numpy.array([31,31,31,31,31,31,31,31,31,31,\
			    31,31,31,15,15,15,15,20,20,15,\
                            10,10,10,31,31,31,31,31,31,31,\
                            31,31,31,31,31,31],dtype=int)
        self.WIalpha=int(self.alpha*self.W.sum()/self.I)
        
        self.maxConstraintsViolation=self.WIalpha*self.I*self.W.sum()

	'''l lumbers corresponding to the elements of the vector Y'''
        IJKmap=numpy.zeros((self.I,self.J,self.Kmax,self.L,3),dtype=int)
        '''Maps actual boolean variables to the matrix'''
        self.booleanMap=numpy.zeros((self.I,self.J,self.Kmax,self.L),dtype=bool)
        for i in range(0,self.I):
    	    for j in range(0,self.J):
    		for k in range(0,self.K[j]):
    		    for l in range(0,self.L):
    			self.booleanMap[i,j,k,l]=self.z[j,l]
    			IJKmap[i,j,k,l,0]=i
    			IJKmap[i,j,k,l,1]=j
    			IJKmap[i,j,k,l,2]=k
    	'''		self.booleanMap[i,j,k,l]=True'''
    	self.booleanMap=self.booleanMap.reshape(self.I*self.J*self.Kmax*self.L)
        IJKmap=IJKmap.reshape(self.I*self.J*self.Kmax*self.L,3)[self.booleanMap,:]
        '''length of boolean vector to operate with '''
        self.booleanLen=self.booleanMap.sum()
        '''array of indexes corresponding to the first and last position in the boolean vector  for matrix Y'''
        self.IJKstart=numpy.zeros((self.I,self.J,self.Kmax),dtype=int)
        self.IJKstart.fill(-1)
        self.IJKstops=numpy.zeros((self.I,self.J,self.Kmax),dtype=int)
        self.IJKstops.fill(-1)
        for ii in range(0,self.booleanLen):
    	    (i,j,k)=IJKmap[ii]
    	    if self.IJKstart[i,j,k]==-1:
    		self.IJKstart[i,j,k]=ii
	    self.IJKstops[i,j,k]=ii+1
        self.IJKstart=filter(lambda x:x>=0,self.IJKstart.reshape(self.I*self.J*self.Kmax))
        self.IJKstops=filter(lambda x:x>=0,self.IJKstops.reshape(self.I*self.J*self.Kmax))
        self.Ncuts=len(self.IJKstart)
        
        
    def C(self,Y):
	'''Y is a  boolean 4D matrix (see the Antamoshkin's paper)'''
	'''Y[i,j,k,l] = Y [days,FB number, FK mumber, production kind]'''
	
	return Y[0,:,:,:].sum()+((1-Y[:-1,:,:,:])*Y[1:,:,:,:]).sum()
	
    def test_C(self):
	y=numpy.array([[[[1]]],[[[0]]],[[[0]]],[[[1]]],[[[1]]],[[[1]]],[[[0]]],[[[1]]],[[[0]]],[[[1]]],[[[1]]]],dtype=bool)
	assert self.C(y)==4
	y=numpy.array([[[[0,1,0,1,1,1,0,1,0,1,1]]]],dtype=bool).reshape(11,1,1,1)
	assert self.C(y)==4
	y=numpy.array([[[[1,1,0,1,1,1,0,1,0,1,1]]]],dtype=bool).reshape(11,1,1,1)
	assert self.C(y)==4
	y=numpy.array([[[[0,0,1,1,0,1,0,1,0,1,0]]]],dtype=bool).reshape(11,1,1,1)
	assert self.C(y)==4
	
	
    def penalty(self,Y):
	'''Function of the 1st constraint'''
	Y1=2+Y[1: ,:,:,:]
	Y2=Y[:-1,:,:,:]
	YProd=Y1*Y2
	penal=0.0
	'''1st, we try to count A2 penalties'''
	'''for the 1st i=0, it is just multiplication by 2'''
	A2_arr=numpy.zeros(self.I,dtype=float)
	WIalfaf=float(self.WIalpha)
	for i in range(0,self.I):
	    A2=0
	    for j in range(0,self.J):
		for k in range(0,self.K[j]):
		    if i==0:
			A2+=(Y[0,j,k]*2*self.V).sum()
		    else:
			i1=i-1
			'''print "i=",i,"j=",j,"k=",k,"Y=",Y1[i1,j,k]*Y2[i1,j,k]'''
			A2+=(YProd[i1,j,k]*self.V).sum()
	    A2_arr[i]=0. if A2>self.WIalpha else float(self.WIalpha-A2)/WIalfaf

	A1_arr=numpy.zeros(self.L,dtype=float)
	for l in range(0,self.L):
	    A1=(Y[0,:,:,l]*2*self.V[l]).sum()+\
		    (YProd[:self.T[l]-1,:,:,l]*self.V[l]).sum()
	    A1_arr[l]=0. if A1>self.W[l] else float(self.W[l]-A1)/float(self.W[l])
	    
	'''Additional constraint'''
	z_nevyp=0
	for i in range(0,self.I):
	    for k in range(0,self.Kmax):
		z_nevyp+=(Y[i,:,k,:]*self.notz).sum()
	'''By now, A1 and A2 are calculated and we have to calculate the penalty itself'''
	'''Additional constraint sum(Y)<=1, constraint 9 in the paper'''
	sumYpenalty=0
	for i in range(0,self.I):
	    for j in range(0,self.J):
		for k in range(0,self.K[j]):
		    sumY=Y[i,j,k].sum()
		    sumYpenalty+=0 if sumY<=1 else sumY-1
	assert z_nevyp==0 
	assert sumYpenalty==0
	return A1_arr.sum()+A2_arr.sum()*0.+float(z_nevyp)/float(self.booleanLen)+sumYpenalty
		
	
    def C_and_penalty(self,X):
	Y=self.scalar_X_transcode(X)
	return (self.C(Y),self.penalty(Y))
    
    def test_penalty(self):
	y=numpy.ones((self.I,self.J,self.Kmax,self.L),dtype=bool)
	for i in range(3,15):
	    y[i]=numpy.zeros((self.J,self.Kmax,self.L),dtype=bool)
	self.penalty(y)
    
    def scalar_X_transcode(self,X):
	'''based on vector X of boolean variables generates and returns matrix Y'''
	YY=numpy.zeros(self.I*self.J*self.Kmax*self.L,dtype=bool)
	YY[self.booleanMap]=X
	return YY.reshape(self.I,self.J,self.Kmax,self.L)
	
    def test_scalar_transcode(self):
	X=numpy.ones(self.booleanLen,dtype=bool)
	Y=self.scalar_X_transcode(X)
	assert Y.sum()==self.booleanLen
	X=numpy.zeros(self.booleanLen,dtype=bool)
	Y=self.scalar_X_transcode(X)
	assert Y.sum()==0
	
	
	