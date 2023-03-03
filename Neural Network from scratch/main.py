from MLP_tools import *

xs = [
    [1.0,3.0,2.0],
    [-3.0,-1.0,0.5],
 
]
ys = [1.0,-1.0] #labels

n = MLP(3,[2,2,1])
n(xs[0])