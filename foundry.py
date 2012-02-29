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
        self.W=numpy.array([20,25000,30,450,45,145,450,4500,4000,400,\
			    22241,12241,940,745,445,20470,12350,145,540,550,\
                            5449,148,140,3543,6441,5545,2249,345,440,549,\
                            140,140,150,145,4598,12000],dtype=int)
        '''Ability of the FB's to produce kinds of production'''
        self.z=numpy.array([[1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1], \
    			    [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0], \
    			    [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 1,1,1,0,0,0]  \
    			    ],dtype=int)
	''' Terms of finishing production'''
	self.T=numpy.array([31,31,31,31,31,31,31,31,31,31,\
			    31,31,31,15,15,15,15, 5, 5,15,\
                            10,10,10,31,31,31,31,31,31,31,\
                            31,31,31,31,31,31],dtype=int)
        
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
	A2_arr=numpy.zeros(self.I,dtype=int)
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
	    print "A2 ",i," = ",A2
	    A2_arr[i]=A2
	A1_arr=numpy.zeros(self.L,dtype=int)
	for l in range(0,self.L):
	    A1=0
	    
		
	
    def test_penalty(self):
	y=numpy.ones((self.I,self.J,self.Kmax,self.L),dtype=bool)
	for i in range(3,15):
	    y[i]=numpy.zeros((self.J,self.Kmax,self.L),dtype=bool)
	self.penalty(y)
    
	