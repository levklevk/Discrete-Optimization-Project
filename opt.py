#!/usr/bin/python
import sys
import numpy
import re


class opt:

    def __init__(self):
	self.dims=0
	self.dimc=0
	self.dimc0=0
	self.dimc_var=0
	self.sum_coef=0
	self.sum_coef2=0
	self.coef1=0


    def inputAll(self):
	'''Reads data from standard input'''
	s=sys.stdin.readline()
	re.sub("\s+" , " ", s)
	ss=s.split()
	self.dims=int(ss[0])
	self.dimc=int(ss[1])
	self.dimc0=int(ss[2])
	self.dimc_var=int(ss[3])
	self.sum_coef=float(ss[4])
	self.sum_coef2=float(ss[5])
	self.a=numpy.zeros(self.dims,dtype=float)
	self.a2=numpy.zeros(self.dims,dtype=float)
	self.bb=numpy.zeros(self.dimc0,dtype=float)
	self.p=numpy.zeros(self.dims,dtype=float)
	self.b=numpy.matrix(numpy.zeros(self.dims*self.dimc0,dtype=float)).reshape(self.dimc0,self.dims)
	for i in range(0,self.dims):
	    s=sys.stdin.readline()
	    re.sub("\s+"," ",s)
	    ss=s.split()
	    self.a[i]=float(ss[0])
	    self.a2[i]=float(ss[1])
	    self.p[i]=float(ss[2])
	for j in range(0,self.dimc0):
	    for i in range(0,self.dims):
		s=sys.stdin.readline()
		re.sub("\s+"," ",s)
		ss=s.split()
		self.b[j,i]=float(ss[0])
	    s=sys.stdin.readline()
	    re.sub("\s+"," ",s)
	    ss=s.split()
	    self.bb[j]=float(ss[0])

    def F(self,x):
	return self.a[x].sum()*(self.sum_coef2-self.a2[x].sum())

    def penalty(self,x):
	sump=0.0
	for j in range(0,self.dimc0):
	    sum_bs=self.b[j,x].sum()
	    if sum_bs>self.bb[j]:
		sump+=abs(sum_bs/self.bb[j])
	return sump*self.dimc0

