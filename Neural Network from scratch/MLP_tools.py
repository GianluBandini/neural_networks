import random
import numpy as np

# Define a class Value
class Value:

    # Initialize the class
    def __init__(self, data, _children = (), _op = '',label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # Define the __repr__ method
    def __repr__(self):
        return f'Value({self.data})'
    
    # Define the __add__ method
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+' )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): # for commutative property, which is called in the case : 2 + a
        return self.__add__(other)
    
    # Define the __mul__ method
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # for commutative property, which is called in the case : 2 * a
        return self.__mul__(other)

    def __pow__(self,other):
        assert isinstance(other, (int, float)), 'The power must be a int ot a float'
        out = Value(self.data ** other, (self,), '**')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out

    # division
    def __truediv__(self, other):
        return self * (other ** -1)
    
    #negation
    def __neg__(self):
        return self * -1
    
    #subtraction
    def __sub__(self,other):
        return self + (-other)
    
    # Define the activation function
    def tanh(self):
        out =  Value(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad = (1 - np.tanh(self.data)**2) * out.grad
        out._backward = _backward
        return out

    # define the exp function
    def exp(self):
        out = Value(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += np.exp(self.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        #build the topological graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self,nin):
        self.w = [Value(np.random.normal(-1,1)) for _ in range(nin)]
        # self.b = Value(np.random.normal(-1,1))
        self.normgains = [Value(np.random.normal(-1,1)) for _ in range(nin)]
        self.normbias= Value(np.random.normal(-1,1))
    
    def __call__(self,x):
        # w * x + b
        wixis = [wi * xi for wi,xi in zip(self.w, x)]
        mean = sum(wixis) * len(wixis)**-1
        print(mean)
        wixis = [item - mean for item in wixis]
        wixis2 = [item ** 2 for item in wixis]
        var = sum(wixis2) * (len(wixis2)-1) ** -1
        var_inv = (var + 1e-6) ** -0.5
        wixis_norm = [item * var_inv for item in wixis]
        wixis_final = [wix_item * gain_item + bias_item for wix_item, gain_item, bias_item in zip(wixis_norm,self.normgains,self.normbias)]
        out = sum(wixis_final)
        return out 
    
    def normalization(self):
        pass
    
    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self,nin,nout):
        self.neurons= [Neuron(nin) for _ in range(nout)]
    
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    
    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]