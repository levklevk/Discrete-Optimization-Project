#!/usr/bin/python
import sys
import numpy
import re
import weber

from pylab import *


class opt:

    def __init__(self):
	self.weber=weber.weber()
	self.dims=self.weber.dims1*self.weber.dims2
	self.dims1=self.weber.dims1
	self.dims2=self.weber.dims2
	self.sum_coef=0
	self.npopulation=0
	self.p_step=0.2
        self.nsteps_noresult=0
        self.nsteps_noresult_after_reset=0
        self.nsteps_samevalue=0
        self.max_steps_noresult=500
        self.max_steps_noresult_after_reset=20
        self.bestresult=(True,-numpy.inf)
        self.bestresult_ever=(True,-numpy.inf)
        self.ncalc=0
        ''' number of access points'''
        self.npoints=self.weber.npoints
        '''the region around the best point which will be adapted by probability'''
        self.adapt_region=10
        self.bestcoords=numpy.zeros((self.npoints,2),dtype=int)
        self.bestcoords.fill(-1)

    def initP(self,p_avg):
	'''initializes p bector'''
	self.p_avg=p_avg
	self.p=numpy.zeros((self.dims1,self.dims2),dtype=float)
	self.p.fill(p_avg)
    
    def inputWeber(self):
	'''sets data given in weber problem'''
	self.p=numpy.zeros((self.dims1,self.dims2),dtype=float)
	self.p.fill(0.5)
	self.sum_coef=1.
	figure(1)
	c=self.weber.customers.copy()
	c=c*(self.weber.obstacles==0)-self.weber.obstacles
	imshow(c)
	savefig("figs/map.jpg")
	
	
	
    def F_P(self,coords):
	self.ncalc+=1
	return(self.weber.C(coords),0.)

    def fill_x_prepare(self):
        ''' Prepares pp array for filling X vector'''
        ''' Counting probabilities that no one variable in each of sets are equal to 1'''
	self.pp1=self.p.sum(1)
	self.pp2=self.p.sum(0)
	self.p_sum=self.pp1.sum()
	self.p_avg=self.p_sum/len(self.pp1)
	''' self.anti_p=1.0-self.p'''
	self.p_copy=self.p.copy()

	    
    def fill_x(self):
	''' fills x with random Boolean values. fill_x_prepare must be already done'''
        ''' used at each step of probability changign algorithm '''
	''' also used for initialization of the genetic algorithm'''
	''' fill everything in accordance with the probabilities p'''
	coords=numpy.zeros((self.npoints,2),dtype=int)
	coords.fill(-1)
	for n in range(0,self.npoints):
	    a=numpy.random.rand()*self.p_sum
	    '''First, we chose the row'''
	    sp=0.
	    for i in range(0,self.dims1):
		sp+=self.pp1[i]
		if sp>=a:
		    '''Row is chosen, now, we try to choose a column'''
		    a=numpy.random.rand()*self.pp1[i]
		    sp2=0.
		    for j in range(0,self.dims2):
			sp2+=self.p_copy[i,j]
			if sp2>=a:
			    self.pp1[i]-=self.p_copy[i,j]
			    self.pp2[j]-=self.p_copy[i,j]
			    self.p_sum-=self.p_copy[i,j]
			    self.p_copy[i,j]=0.0
			    coords[n,0]=i
			    coords[n,1]=j
			    break
		    break
	assert (coords>=0).all()
	return coords
	
		    
		    
	    
    def test_fill_x(self):
	return True

    def calc_sorting(self,xmatrix):
	'''Based on given matrix xmatrix (its rows are the exemplars of the X vector),'''
	'''this function calculates the rerults of estimating the penalty function'''
	'''and the objective function'''
	'''Then, it sorts the results and returns an array of the indexes of'''
	'''all the X's sorted from the best to the worst'''
	'''the results of estimating the FF and FF_modified functions are stored in the'''
	'''arrays results[], nopenalty[] and results_mod[]'''
	'''nopenalty is a Boolean array, each elements  means that we have an exact solution'''
	self.results=numpy.zeros(self.npopulation, dtype=[('ind', int), ('f', float),('penalty',bool),('f_mod',float)])
	for i in range(0,self.npopulation):
	    (res,pen)=self.F_P(xmatrix[i])
	    self.results[i]=(i,res,pen!=0.0,-(res-pen))
	''' By now, we have all the results, modified and pure'''
	''' Let's sort the penalties first, and if there are any pure results'''
	''' sort all the results that those without penalty are in the beginning'''
	self.results.sort(order=["f_mod"])
	'''By now, we have proper order of the results in the 'ind' column of the results'''

    def test_calc_sorting(self):
	x=self.test_fill_x()
	self.calc_sorting(x)
	assert self.results[0][2]<self.results[self.npopulation-1][2]  \
	    or (self.results[0][2]==self.results[self.npopulation-1][2] \
	    and self.results[0][3]<=self.results[self.npopulation-1][3])
    
    def adaptation(self,bestrow,worstrow):
	'''Adapts the probability matrix depending on the results of calc_sorting'''
	'''best and worst rows are the lists of coordinates '''
	'''Probabilities are not in interval [0:1] any more, the are just numbers...'''

	'''At the 1st step, we increase the probabilityes around the "ones" in the best list'''
	'''Then, we decrease the probabilityes around the "ones" in the worst list'''
	''' if the bests and the wersts are the same, the result will be the same as if nothing happened'''
	antistep=1.0/(1.0+self.p_step)
	self.ad_best(bestrow,  1.)
	self.ad_best(worstrow,-1.)


    def ad_best(self,row,coef):
	for e in row:
	    x=e[0]
	    y=e[1]
	    self.ad2(x,y,coef)
	    
    def ad2(self,x,y,coef):
	    ar_coef=1./self.adapt_region
	    for i1 in range(max(0,x-self.adapt_region),min(self.dims1,x+self.adapt_region)):
		for i2 in range(max(0,y-self.adapt_region),min(self.dims2,y+self.adapt_region)):
		    '''distance between the point and its neighbor'''
		    dist=self.adapt_region-max(abs(i1-x),abs(i2-y))
		    p_step1=self.p_step*dist*coef*ar_coef
		    if p_step1>0:
    			self.p[i1,i2]*=(1.0+p_step1)
    		    else:
    			self.p[i1,i2]/=(1.0-p_step1)
		    

    def test_adaptation(self):
	self.npopulation=5
	x=self.test_fill_x()
	self.calc_sorting(x)
	pp=self.p.copy()
	self.adaptation(x[self.results[0][0]],x[self.results[0][0]])
	'''check that nothing has been changed if vectors are equal'''
	assert (self.p==pp).all()
	self.adaptation(x[self.results[0][0]],x[self.results[self.npopulation-1][0]])
	bestx=x[self.results[0][0]]
	worstx=x[self.results[self.npopulation-1][0]]
	'''check that if elements of some pair in (xbest,xworst) are not equal then p is changed for this element'''
	assert (self.p[bestx!=worstx]!=pp[bestx!=worstx]).all()
	self.adaptation(x[self.results[0][0]],x[self.results[0][0]])
	
    def possible_reset(self):
	if self.nsteps_noresult_after_reset>self.max_steps_noresult_after_reset :
	    '''We need a reset, total or partial'''
	    self.max_steps_noresult_after_reset+=self.max_steps_noresult_after_reset/5
    	    part=numpy.random.rand()*1.5
	    '''total percent of the variables being reset'''
	    if part>=1.0:
		self.p.fill(self.p_avg)
	    else:
		''' If the reset is partial'''
		rands=numpy.random.rand(self.dims1,self.dims2)
		self.p[rands>part]=self.p_avg
		self.p[rands<=part]=map(lambda x:(x+self.p_avg)*0.5,self.p[rands<=part])
	    '''now, we have to go to the average probability in each set of variables'''
	    self.nsteps_noresult_after_reset=0
	    self.bestresult=(True,-numpy.inf)
	    self.nsteps_samevalue=0
	
	
    def mutationprobchang(self):
	'''It is a dummy function now'''
	nmuts=int(0.5+self.nsteps_samevalue+numpy.random.rand())
	for i in range(0,nmuts):
	    num1=numpy.random.randint(self.dims1)
	    num2=numpy.random.randint(self.dims2)
	    coef=float(numpy.random.randint(2)*2-1)
	    self.ad2(num1,num2,coef)
	
	
    def do_adaptation_strategy(self,xmatrix):
	'''We can chose best and worst points or 2 arbitrary chosen points'''
	'''to prepare the adaptation'''
	'''also, we can perform several steps'''
	if self.results[0][2]==self.results[self.npopulation-1][2] \
	    and self.results[0][3]==self.results[self.npopulation-1][3]:
	    '''If we have absolutely the same results'''
	    self.nsteps_samevalue+=1
	    return
        self.nsteps_samevalue=0
	'''Strategy is so that First, we try to chose the points arbitrary'''
	rnd=(numpy.random.rand(self.npopulation/5,2))*self.npopulation/5
	'''How many times do we do the adaptation'''
	'''But at leas one time'''
	for i in rnd:
	    n1=int(i[0])
	    n2=self.npopulation-int(i[1])-1
	    if self.results[n1][3]!=self.results[n2][3]:
		'''If the values are not the same'''
		self.adaptation(xmatrix[self.results[n1][0]],xmatrix[self.results[n2][0]])
	''' And, in the end, we perform step with the best and the worst values'''
	self.adaptation(xmatrix[self.results[0][0]],xmatrix[self.results[self.npopulation-1][0]])
	'''Now.we calculate the average probability'''
	self.p_avg=self.p.mean()
	
		
    def checkResults(self,x):
	'''Checks the results calculated and counts the the results which are not good'''
	self.nsteps_noresult+=1
	self.nsteps_noresult_after_reset+=1
	if  self.bestresult[1]<(-self.results[0][3]):
	    self.nsteps_noresult_after_reset=0
	    self.nsteps_sameresult=0
	    self.bestresult=(self.results[0][2],-self.results[0][3])
	if self.bestresult_ever[0]>self.results[0][2] \
	    or (self.bestresult_ever[0]==self.results[0][2]\
	    and self.bestresult_ever[1]<(-self.results[0][3])):
    		self.nsteps_noresult=0
    		self.nsteps_sameresult=0
    		self.nsteps_noresult_after_reset=0
		self.bestresult_ever=(self.results[0][2],-self.results[0][3])
		self.bestcoords=x[self.results[0][0]]
		figure(1)
		recv=self.weber.C(x[self.results[0][0]],True)
		for i in x[self.results[0][0]]:
		    recv[i[0],i[1]]=100.
		imshow(recv)
		savefig("figs/recv_"+str(self.npoints)+"_"+str(self.ncalc)+".jpg")
		
		
	
    def perform1stepprobchang(self):
	'''performs 1 step of probability changing algorithm'''
	x=numpy.zeros((self.npopulation,self.npoints,2),dtype=int)
	for j in range(0,self.npopulation):
    	    self.fill_x_prepare()
	    x[j]=self.fill_x()
	self.calc_sorting(x)
	self.do_adaptation_strategy(x)
	self.checkResults(x)
	self.mutationprobchang()
	self.possible_reset()
    

def main1(npoints):
    o=opt()
    p_start=0.5
    o.inputWeber()
       
    o.npopulation=int(numpy.sqrt(o.dims1+o.dims2))
    o.initP(p_start)
    nsteps=0
    print "N_points=",npoints
    o.npoints=npoints
    while o.ncalc<10000:
	o.perform1stepprobchang()
        nsteps+=1
	print "Ncalculations=",o.ncalc,", bestResultEver=",o.bestresult_ever[1],", conditionsSatisfied=",not o.bestresult_ever[0],", bestResultSinceRestart=",o.bestresult[1],", currentBestResult=",-o.results[0][3],", currentWorstResult=",-o.results[-1][3],", p_avg=",o.p_avg,", StepsWithoutResults=",o.nsteps_noresult_after_reset
	if (nsteps==2 and o.ncalc<40) or nsteps==10:
	    nsteps=0
	    print "Best coordinates:"
	    print o.bestcoords
	    figure(1)
	    P=log(o.p)
	    imshow(P,interpolation='nearest',cmap='gray')
	    savefig("figs/web"+str(o.npoints)+"_"+str(o.ncalc)+".jpg")
    print "Best=",o.bestresult_ever[1],",ConditionsSatisfied=",not o.bestresult_ever[0],", p_start=",p_start
    print "Best coordinates ever:"
    print o.bestcoords

def main():
    for i in range(6,12):
	main1(i)


