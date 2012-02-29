#!/usr/bin/python
import sys
import numpy
import re
import foundry


class opt:

    def __init__(self):
	self.dims=0
	self.sum_coef=0
	self.npopulation=0
	self.p_step=0.3
        self.nsteps_noresult=0
        self.nsteps_noresult_after_reset=0
        self.nsteps_samevalue=0
        self.max_steps_noresult=10000
        self.max_steps_noresult_after_reset=30
        self.bestresult=(True,-numpy.inf)
        self.bestresult_ever=(True,-numpy.inf)
        self.ncalc=0
        self.foundry=foundry.foundry()

    def initP(self,p_avg):
	'''initializes p bector'''
	self.p_avg=p_avg
	self.p=numpy.zeros(self.dims,dtype=float)
	self.p.fill(p_avg)
    
    def inputFoundry(self):
	'''sets data given in foundry problem'''
	self.dims=self.foundry.booleanLen
	self.p=numpy.zeros(self.dims,dtype=float)
	self.sum_coef=self.foundry.booleanLen
	
    def F_P(self,x):
	self.ncalc+=1
	tupl=self.foundry.C_and_penalty(x)
	return(-tupl[0],tupl[1]*self.sum_coef)

    def fill_x_prepare(self):
        ''' Prepares pp array for filling X vector'''
        ''' Counting probabilities that no one variable in each of sets are equal to 1'''
	self.pp =numpy.zeros(self.foundry.Ncuts,dtype=float)
	self.pp2=numpy.zeros(self.foundry.Ncuts,dtype=float)
	''' self.anti_p=1.0-self.p'''
	for i in range(0,self.foundry.Ncuts):
	    idvar =self.foundry.IJKstart[i]
	    idvar2=self.foundry.IJKstops[i]
	    ''' mult_p=self.anti_p[idvar:idvar2].prod()'''
	    sum_p =self.p[idvar:idvar2].sum()
	    ''' self.pp[i]=mult_p+sum_p'''
	    ''' probability that all variables == 0 '''
	    self.pp[i] =1.0-sum_p/(idvar2-idvar)
	    '''	total probability    '''
	    self.pp2[i]=self.pp[i]+sum_p

	    
    def fill_x(self):
	''' fills x with random Boolean values. fill_x_prepare must be already done'''
        ''' used at each step of probability changign algorithm '''
	''' also used for initialization of the genetic algorithm'''
	rands=numpy.random.rand(self.foundry.Ncuts)
	x=numpy.zeros(self.dims,dtype=bool)
	for i in range(0,self.foundry.Ncuts):
	    ppp=self.pp2[i]*rands[i]
	    p0=0.0
	    idvar =self.foundry.IJKstart[i]
	    idvar2=self.foundry.IJKstops[i]
	    n_el=True
	    for k in range(idvar,idvar2):
		p0+=self.p[k]
		if p0>ppp and n_el:
		    n_el=False
		    x[k]=True
		    break
	return x
	
	
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
	self.npopulation=5
	x=self.test_fill_x()
	self.calc_sorting(x)
	assert self.results[0][2]<self.results[self.npopulation-1][2]  \
	    or (self.results[0][2]==self.results[self.npopulation-1][2] \
	    and self.results[0][3]<=self.results[self.npopulation-1][3])
    
    def adaptation(self,bestrow,worstrow):
	'''Adapts the probability vector depending on the results of calc_sorting'''
	'''best and worst rows are the rows of the X array '''

	'1st, compose list of array indices to change'
	antistep=1.0/(1.0+self.p_step)
	for i in numpy.array(range(0,self.dims))[bestrow!=worstrow]:
	    '''check if the worst if less than the best elment'''
	    if bestrow[i]>worstrow[i]:
        	if self.p[i] < 0.5 :
		    self.p[i]=self.p[i]*(1.0+self.p_step)
                else:
		    self.p[i]+=(1.0-self.p[i])*self.p_step
	    else:
		if self.p[i] < 0.5:
		    self.p[i]=self.p[i]*antistep
		else:
		    self.p[i]-=(1.0-self.p[i])*self.p_step


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
    	    part=numpy.random.rand()*1.5
	    '''total percent of the variables being reset'''
	    if part>=1.0:
		self.p.fill(self.p_avg)
	    else:
		''' If the reset is partial'''
		rands=numpy.random.rand(self.dims)
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
	    num=numpy.random.randint(self.dims)
	    self.p[num]=self.p_avg
	
	
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
	rnd=numpy.random.rand(self.npopulation/2,2)
	'''How many times do we do the adaptation'''
	'''But at leas one time'''
	rnd=[]
	for i in rnd:
	    n1=max(i[0],i[1])
	    n2=min(i[0],i[1])
	    if self.results[n1][3]!=self.results[n2][3]:
		'''If the values are not the same'''
		self.adaptation(xmatrix[results[n1][0]],xmatrix[results[n2][0]])
	''' And, in the end, we perform step with the best and the worst values'''
	self.adaptation(xmatrix[self.results[0][0]],xmatrix[self.results[self.npopulation-1][0]])
	'''Now.we calculate the average probability'''
	self.p_avg=self.p.mean()
	
		
    def checkResults(self):
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
	    
	    
	    
	
    def perform1stepprobchang(self):
	'''performs 1 step of probability changing algorithm'''
	x=numpy.zeros((self.npopulation,self.dims),dtype=bool)
	self.fill_x_prepare()
	for j in range(0,self.npopulation):
	    x[j]=self.fill_x()
	self.calc_sorting(x)
	self.do_adaptation_strategy(x)
	self.checkResults()
	self.mutationprobchang()
	self.possible_reset()
    

                                 

def main1(p_start):
    o=opt()
    o.inputFoundry()
       
    o.npopulation=int(4+numpy.sqrt(o.dims))
    o.initP(p_start)
    x=numpy.zeros(o.dims,dtype=bool)
    x=numpy.zeros((o.npopulation,o.dims),dtype=bool)
    nsteps=0
    print "P_start=",p_start
    while o.ncalc<500000:
	o.perform1stepprobchang()
        nsteps+=1
	if nsteps==1:
	    print "Ncalculations=",o.ncalc,", bestResultEver=",o.bestresult_ever[1],", conditionsSatisfied=",not o.bestresult_ever[0],", bestResultSinceRestart=",o.bestresult[1],", currentBestResult=",-o.results[0][3],", currentWorstResult=",-o.results[-1][3],", p_avg=",o.p_avg,", StepsWithoutResults=",o.nsteps_noresult_after_reset
	nsteps=0
    print "Best=",o.bestresult_ever[1],",ConditionsSatisfied=",not o.bestresult_ever[0],", p_start=",p_start

def main():
    for i in numpy.arange(0.3,0.9,0.1):
	main1(i)


