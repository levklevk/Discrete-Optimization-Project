import opt
import numpy
    
def test_F():
    x=numpy.array([True,False,False,True],dtype=bool)
    oo=opt.opt()
    oo.a=numpy.array([1,2,3,4],dtype=float)
    oo.a2=numpy.array([11,12,13,14],dtype=float)
    oo.sum_coef2=50.0
    assert oo.F(x)==5.0*(50.0-25.0)
    
def test_penalty():
    x=numpy.array([True,False,False,True],dtype=bool)
    oo=opt.opt()
    oo.b=numpy.matrix([[1,2,3,4],[10,11,12,13]],dtype=float)
    oo.bb=numpy.array([6,3],dtype=float)
    oo.dimc0=2
    assert abs(oo.penalty(x)-15.3333333333)<0.0001



test_F()
test_penalty()