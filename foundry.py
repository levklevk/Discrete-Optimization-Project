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
	'''Number of orders and kinds of production'''
	self.L=36
	''' Rigidity coeefficient'''
	self.alpha=0.5
	'''Productivities for change'''
	self.V=numpy.array([40,40,40,45,45,45,50,45,40,40,\
			    41,41,40,45,45,47,50,45,40,50,\
                            49,48,40,43,41,45,49,45,40,49,\
                            40,40,50,45,45],dtype=int)
        ''' Vilumes of orders'''
        self.W=numpy.array([20,25000,30,450,45,145,450,4500,4000,400,\
			    22241,12241,940,745,445,20470,12350,145,540,550,\
                            5449,148,140,3543,6441,5545,2249,345,440,549,\
                            140,140,150,145,4598],dtype=int)
        '''Ability of the FB's to produce kinds of production'''
        self z=numpy.array([[1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1],\
    			    [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0],\
    			    [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 1,1,1,0,0,0],\
    			    ],dtype=int)
	''' Terms of finishing production'''
	self T=numpy.array([31,31,31,31,31,31,31,31,31,31,\
			    31,31,31,15,15,15,15, 5, 5,15,\
                            10,10,10,31,31,31,31,31,31,31,\
                            31,31,31,31,31],dtype=int)
        
    def C(self,Y):
	'''Y is a  boolean 4D matrix (see the Antamoshkin's paper)'''
	'''Y[i,j,k,l] = Y [days,FB number, FK mumber, production kind]'''
	return Y[0]+((1-Y[:-1])*Y[1:]).sum()
	
    def test_C(self)
	y=numpy.array([1,0,0,1,1,1,0,1,0,1,1],dtype=bool)
	assert self.C(y)==4
	y=numpy.array([0,1,0,1,1,1,0,1,0,1,1],dtype=bool)
	assert self.C(y)==4
	y=numpy.array([1,1,0,1,1,1,0,1,0,1,1],dtype=bool)
	assert self.C(y)==4
	y=numpy.array([0,0,1,1,0,1,0,1,0,1,0],dtype=bool)
	assert self.C(y)==4
	
	
    