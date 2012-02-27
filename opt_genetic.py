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
	self.npopulation=0
	self.p_step=0.3
        self.nsteps_noresult=0
        self.nsteps_noresult_after_reset=0
        self.nsteps_samevalue=0
        self.max_steps_noresult=10000
        self.max_steps_noresult_after_reset=100
        self.bestresult=(False,-numpy.inf)
        self.bestresult_ever=(False,-numpy.inf)
        self.ncalc=0
        self.best_x_ever=[]
        self.best_x_after_restart=[]
        self.genetic_1stgeneration=True
        

    def initP(self,p_avg):
	'''initializes p bector'''
	self.p_avg=p_avg
	self.p=numpy.zeros(self.dims,dtype=float)
	self.p.fill(p_avg)
	self.sum_coef =abs(self.a).sum()
	self.sum_coef2=abs(self.a2).sum()
    
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
	self.ncalc+=1
	return self.a[x].sum()*(self.sum_coef2-self.a2[x].sum())

    def penalty(self,x):
	'''returns penalty value'''
	sump=0.0
	for j in range(0,self.dimc0):
	    sum_bs=self.b[j,x].sum()
	    if sum_bs>self.bb[j]:
		sump+=abs((sum_bs-self.bb[j])/self.bb[j])
	return sump*self.dimc0*self.dims*self.sum_coef
    
    def F_mod(self,x):
	return self.F(x)-self.penalty(x)

    def fill_x_prepare(self):
        ''' Prepares pp array for filling X vector'''
	self.pp=numpy.zeros(self.dims/self.dimc_var,dtype=float)
	self.anti_p=1.0-self.p
	for i in range(0,self.dims/self.dimc_var):
	    idvar=i*self.dimc_var
	    mult_p=self.anti_p[idvar:idvar+self.dimc_var].prod();
	    sum_p =self.p[idvar:idvar+self.dimc_var].sum();
	    self.pp[i]=mult_p+sum_p
	    
    def fill_x(self):
	''' fills x wwith random Boolean values. fill_x_prepare must be already done'''
        ''' used at each step of probability changign algorithm '''
	''' also used for initialization of the genetic algorithm'''
	rands=numpy.random.rand(self.dims/self.dimc_var)
	empty_x=numpy.zeros(self.dimc_var,dtype=bool)
	x=numpy.zeros(self.dims,dtype=bool)
	for i in range(0,self.dims/self.dimc_var):
	    ppp=self.pp[i]*rands[i]
	    p0=0.0
	    idvar=i*self.dimc_var
	    n_el=True
	    for k in range(0,self.dimc_var):
		p0+=self.p[idvar+k]
		if p0>ppp and n_el:
		    n_el=False
		    x[idvar+k]=True
		    break
	return x
	
	
    def test_fill_x(self):
	x=numpy.zeros(self.dims*self.npopulation,dtype=bool).reshape(self.npopulation,self.dims)
	self.fill_x_prepare()
	for i in range(0,self.npopulation):
	    x[i]=self.fill_x()
	    for ii in range(0,self.dims/self.dimc_var):
		su=0
		for k in range(0,self.dimc_var):
		    su+=0+x[i,ii*self.dimc_var+k]
		assert su<=1
	return x


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
	    res=self.F(xmatrix[i])
	    pen=self.penalty(xmatrix[i])
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
	changedsegs=[]
	prev=-1
	for i in numpy.array(range(0,self.dims))[bestrow!=worstrow]:
	    nsec=i/self.dimc_var
	    if prev!=nsec:
    		changedsegs.append(nsec)
	    prev=nsec
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
	''' if we search by groups of variables (k vars in each grp,
           then it is better if the summary probability of all variants
           of position Xk=1 in each group
           plus probebility that all the Xk=0
          are equal to 1'''
        dim=self.dimc_var
	for i in changedsegs:
	    idvars=i*self.dimc_var
	    ppp0=(1.0-self.p[idvars:idvars+dim]).prod()
	    sumprob=self.p[idvars:idvars+dim].sum()
	    ppp0=1.0/(ppp0+sumprob)
	    self.p[idvars:idvars+dim]*=ppp0


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
    	    dim=self.dimc_var
	    for i in range(0,self.dims/self.dimc_var):
		idvars=i*self.dimc_var
    		ppp0=(1.0-self.p[idvars:idvars+dim]).prod()
		sumprob=self.p[idvars:idvars+dim].sum()
		ppp0=1.0/(ppp0+sumprob)
		self.p[idvars:idvars+dim]*=ppp0
	    self.nsteps_noresult_after_reset=0
	    self.bestresult=(False,-numpy.inf)
	    self.nsteps_samevalue=0
	
	
    def mutationprobchang(self):
	'''It is a dummy function now'''
	changedsegs=[]
	nmuts=int(0.5+self.nsteps_samevalue+numpy.random.rand())
	for i in range(0,nmuts):
	    num=numpy.random.randint(self.dims)
	    self.p[num]=self.p_avg
	    changedsegs.append(i/self.dimc_var)
        dim=self.dimc_var
	for i in changedsegs:
	    idvars=i*self.dimc_var
	    ppp0=(1.0-self.p[idvars:idvars+dim]).prod()
	    sumprob=self.p[idvars:idvars+dim].sum()
	    ppp0=1.0/(ppp0+sumprob)
	    self.p[idvars:idvars+dim]*=ppp0
	
	
	
	
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
	
		
    def checkResults(self,xmatrix):
	'''Checks the results calculated and counts the the results which are not good'''
	self.nsteps_noresult+=1
	self.nsteps_noresult_after_reset+=1
	if  self.bestresult[1]<(-self.results[0][3]):
	    self.nsteps_noresult_after_reset=0
	    self.nsteps_sameresult=0
	    self.bestresult=(self.results[0][2],-self.results[0][3])
	    self.best_x_after_restart=xmatrix[self.results[0][0]]
	if self.bestresult_ever[0]<self.results[0][2] \
	    or (self.bestresult_ever[0]==self.results[0][2]\
	    and self.bestresult_ever[1]<(-self.results[0][3])):
    		self.nsteps_noresult=0
    		self.nsteps_sameresult=0
    		self.nsteps_noresult_after_reset=0
		self.bestresult_ever=(self.results[0][2],-self.results[0][3])
		self.best_x_ever=xmatrix[self.results[0][0]]
	    
	    
	    
	
    def perform1stepprobchang(self):
	'''performs 1 step of probability changing algorithm'''
	x=numpy.zeros((self.npopulation,self.dims),dtype=bool)
	self.fill_x_prepare()
	for j in range(0,self.npopulation):
	    x[j]=self.fill_x()
	self.calc_sorting(x)
	self.do_adaptation_strategy(x)
	self.checkResults(x)
	self.mutationprobchang()
	self.possible_reset()
    
    def gen_set(self,dims,dimc_var,dimc):
	'''Random testing set generator'''
	self.sum_coef=0.0
	self.sum_coef2=0.0
	self.dimc=dimc
	self.dimc0=dimc
	self.dims=dims
	self.dimc_var=dimc_var
	self.a=numpy.zeros(dims,dtype=float)
	self.a2=self.a.copy()
	self.bb=numpy.zeros(dimc,dtype=float)
	self.b=numpy.zeros(dims*dimc,dtype=float).reshape(dimc,dims)
	
	for i in range(0,dims):
	    self.a[i]=self.coef()
	    self.a2[i]=self.coef()*max(numpy.random.rand()-0.1,0.0)
	for j in range(0,dimc,2):
	    sumb=0.0
	    for i in range(0,dims):
	        self.b[j,i]=self.coef()*max(numpy.random.rand()-0.2,0.0)
	        self.b[j+1,i]=-self.b[j,i]
		sumb+=self.b[j,i]
    	    self.bb[j]=sumb*(0.35+numpy.random.rand()*0.35)/dimc_var
    	    self.bb[j+1]=-self.bb[j]*(0.1+numpy.random.rand()*0.55)/dimc_var
	                
		                           
	
	

    def coef(self):
	'''Additional function, useb by random set generator'''
	return 1.0/(numpy.random.rand()+0.1)
	
    def genetic_adaptation(self,xbest,xworst,num_best,num_worst):
	''' Performs adaptation in genetic style, returns a new exemplar generated'''
	num_replaced=numpy.random.randint(self.dims+1)*(self.npopulation-num_worst)/(self.npopulation-num_best+self.npopulation-num_worst)
	num_replaced=min(num_replaced,self.dims-1)
	num_replaced=max(num_replaced,1)
	print "replace",num_replaced
	'''at least one will be replaced'''
	first_replaced=numpy.random.randint(self.dims)
	'''Calculate.how many items we replace'''
	newitem=xbest.copy()
	changedsets=[]
	if first_replaced%self.dimc_var:
	    changedsets.append(first_replaced/self.dimc_var)
	if first_replaced+num_replaced<=self.dims:
	    newitem[first_replaced:first_replaced+num_replaced]=xworst[first_replaced:first_replaced+num_replaced]
	    if (first_replaced+num_replaced)%self.dimc_var:
		changedsets.append((first_replaced+num_replaced)/self.dimc_var)
	else:
	    newitem[first_replaced:self.dims]=xworst[first_replaced:self.dims]
	    if (first_replaced+num_replaced-self.dims)%self.dimc_var:
		changedsets.append((first_replaced+num_replaced-self.dims)/self.dimc_var)
	    newitem[0:first_replaced+num_replaced-self.dims]=xworst[0:first_replaced+num_replaced-self.dims]
	    '''here, we perform checking that on the borders there are no duplication in x's generated'''
	for i in changedsets:
	    '''here, we check that tere is no situation when there are 2 true in the same subset of variables'''
	    k=newitem[i*self.dimc_var:i*self.dimc_var+self.dimc_var].sum()
	    if k>1:
		if numpy.random.randint(3):
		    newitem[i*self.dimc_var:i*self.dimc_var+self.dimc_var]=xbest[i*self.dimc_var:i*self.dimc_var+self.dimc_var]
		else:
		    newitem[i*self.dimc_var:i*self.dimc_var+self.dimc_var]=xworst[i*self.dimc_var:i*self.dimc_var+self.dimc_var]
    		''' But now, we try to put this "one" element to any other place!!!'''
    		tryset=numpy.random.randint(self.dims/self.dimc_var)
    		k=newitem[tryset*self.dimc_var:tryset*self.dimc_var+self.dimc_var].sum()
    		if (k==0):
    		    newitem[tryset*self.dimc_var+numpy.random.randint(self.dimc_var)]=True
	'''Here, we perform possible mutation'''
	while numpy.random.randint(100)==0:
	    nummut=numpy.random.randint(self.dims)
	    newitem[nummut]=not newitem[nummut]
	return newitem

    def genetic_1st_generation(self):
	'''Generate the 1st generation in accordance with the p_avg'''
	xmatrix=numpy.zeros(self.npopulation*self.dims,dtype=bool).reshape(self.npopulation,self.dims)
	self.p.fill(self.p_avg)
	self.fill_x_prepare()
	for i in range(0,self.npopulation):
	    xmatrix[i]=self.fill_x()
	return xmatrix
	    
    def genetic_generation(self,oldgen):
	'''generates the next generation based on the previous'''
	if self.genetic_1stgeneration:
	    xmatrix=self.genetic_1st_generation()
	    self.genetic_1stgeneration=False
	else:
	    xmatrix=numpy.zeros(self.npopulation*self.dims,dtype=bool).reshape(self.npopulation,self.dims)
    	    self.calc_sorting(oldgen)
	    self.checkResults(oldgen)
	    ''' now, we have some results '''
	    elite=int(self.npopulation/4)
    	    noelite=self.npopulation-elite
	    for i in range(0,self.npopulation):
		el2=numpy.random.randint(elite)+1
		el1=numpy.random.randint(el2)
		xmatrix[i]=self.genetic_adaptation(oldgen[el1],oldgen[el2],el1,el2)
    	    self.genetic_possible_reset()
	return xmatrix
	    
    def genetic_possible_reset(self):
	'''Resets if there were no result for too long time'''
	if self.nsteps_noresult_after_reset>self.max_steps_noresult_after_reset :
	    self.p_avg=(0.0+self.best_x_ever.sum())/self.dims
	    self.genetic_1stgeneration=True

o=opt()
o.inputAll()
   
'''o.gen_set(8,2,4)
print "a= ",o.a
print "a2=",o.a2
print "b=",o.b
print "bb=",o.bb
'''
o.npopulation=int(4+numpy.sqrt(o.dims))
o.initP(0.3/o.dimc_var)
x=numpy.zeros((o.npopulation,o.dims),dtype=bool)
nsteps=0
x=o.genetic_1st_generation()
print x
while o.ncalc<10000:
    x=o.genetic_generation(x)
    nsteps+=1
    if nsteps==10:
	print "steps:",o.ncalc,", result=",o.bestresult_ever[1],o.bestresult_ever[0]
	nsteps=0
print "Best=",o.bestresult_ever[1],o.bestresult_ever[0]
