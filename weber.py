#!/usr/bin/python
'''Module containing the objective function for weber in discrete coordinates optimization'''
''''''
import sys
import numpy
import re


class weber(object):

    def __init__(self):
	''' Instead of Boolean variables, we must operate arrays of coordinates'''
	''' of the points where the boolean variables are non-zero'''
	'''Number of access points placed'''
	self.npoints=10
	'''Number of lines'''
	self.dims1=100
	'''Number of coumns'''
	self.dims2=50
	''' total=100x50 =5000 Boolean variables'''
	self.customers=numpy.zeros((self.dims1,self.dims2),dtype=float)

        ''' Now.we fill some regions of customers with their cost values'''
        self.customers[5:10,10:20].fill(3.)
        self.customers[7:9, 30:35].fill(4.)
        self.customers[7:9, 11:13].fill(5.)
        self.customers[20:60,5:45].fill(1.)
        self.customers[30:40,3:25].fill(5.)
        self.customers[55:80,2:35].fill(7.)
       
        ''' The obstacles which interfere the signal propagation '''
        self.obstacles=numpy.zeros((self.dims1,self.dims2),dtype=float)
        self.obstacles[20:70,20:21].fill(10.)
        self.obstacles[69:70,20:35].fill(20.)
        self.obstacles[10:15,5:45].fill(4.)
       
        ''' A distance (in squares) after which the path loss is +Infinity to avoid too much complicated computation'''
        self.critical_distance=30
       
        ''' How many meters are there in one square '''
        self.MetersPerSquare=2.0
        
        '''Level of signal which is considered as critical'''
        self.minSignalLevel=10.0
       
       
    def signalLevel(self,x1,y1,x2,y2):
	'''Returns signal level in range 0:100'''
	''' First, we calculate the real distance '''
	xx=abs(x1-x2)
	yy=abs(y2-y1)
	if xx>self.critical_distance or yy>self.critical_distance:
	    return 0
	
	D=self.MetersPerSquare*numpy.sqrt(xx*xx+yy*yy)
	if D<1.0:
	    D=1.
	FreeSpaceLoss=40.0+20.0*numpy.log10(D)
	'''Now. we must sum all the obstacles along the path'''
	'''first, we calculate the path'''
	if xx<=1 and yy<=1:
	    return FreeSpaceLoss
	''' If it is neighbouring points, no problem with additional path loss'''
	minx=min(x1,x2)
	miny=min(y1,y2) 
	maxx=max(x1,x2)
	maxy=max(y1,y2)
	yrang=maxy-miny
	xrang=maxx-minx
	diag=(x1==minx and y2==miny) or (x2==minx and y1==miny)
	addloss=0
	o=self.obstacles
	if not diag:
	    if (xx>yy):
		''' Petform steps with x'''
    		for x in range(minx+1,maxx):
		    y=miny+(x-minx)*yrang/xrang
		    addloss+=o[x,y]
	    else:
		''' Petform steps with y'''
		for y in range(miny+1,maxy):
		    x=minx+(y-miny)*xrang/yrang
		    addloss+=o[x,y]
	else:
	    if (xx>yy):
		''' Petform steps with x'''
    		for x in range(minx+1,maxx):
		    y=maxy-(x-minx)*yrang/xrang
		    addloss+=o[x,y]
	    else:
		''' Petform steps with y'''
		for y in range(miny+1,maxy):
		    x=maxx-(y-miny)*xrang/yrang
		    addloss+=o[x,y]
    
	return max(76.-(FreeSpaceLoss+addloss),0.)
	

    ''' Objective function '''
    def C(self,Coords,return_matrix=False):
	'''Actually, it is sum of maximum signal level recieved by every customer'''
	
	'''Coords is array of coordinates (2d), 2nd dimension is [0:1]'''
	''' if come customer doesn't receive signal at at least critical level, it adds -its '''
	'''weight*5 for each unit of signal up to critical level its negative weight to the result'''
        ''' first, we construct new matrix of signals received'''
        recv=numpy.zeros((self.dims1,self.dims2),dtype=float)
        '''Now. we scan all the emitters'''
        for i in range(0,len(Coords)):
    	    ex=Coords[i,0]
    	    ey=Coords[i,1]
    	    ''' Now, we scan all points in the region close to the emitter'''
    	    cd=self.critical_distance
    	    for x in range(max(0,ex-cd),min(self.dims1,ex+cd)):
    		for y in range(max(0,ey-cd),min(self.dims2,ey+cd)):
    		    '''Now, we calculate the signal level at this point'''
    		    if self.customers[x,y]!=0.:
        		    recv[x,y]=max(recv[x,y],self.signalLevel(ex,ey,x,y))
    	    '''Now, we calculate the weighted signal levels and penalties'''
    	recv=(recv-self.minSignalLevel)*self.customers
    	if return_matrix:
    	    return recv
    	return (recv>=0.).sum()-5.*(recv<0.).sum()
