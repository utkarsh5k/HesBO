import torch
import numpy as np

class Branin():
    '''
    Takes in an n x 2 input matrix where each row is an observation of dimension 2.
    Outputs n x 1 output matrix where each output has dimension 1.
    '''
    
    def __init__(self, noise_var=0):
        self.domain = np.array([[-5,10],
                             [0,15]])
        self.param = {
            'a':1,
            'b':5.1/(4*math.pi**2),
            'c':5/math.pi,
            'r':6,
            's':10,
            't':1/(8*math.pi)
        }
        
        self.noise_var = noise_var
        self.input_dim = 2
    
    def evaluate(self, x):
        a = self.param['a']
        b = self.param['b']
        c = self.param['c']
        r = self.param['r']
        s = self.param['s']
        t = self.param['t']
        
        f = a*(x[:,1] - b*x[:,1]**2 + c*x[:,0] - r)**2 + s*(1-t)*torch.cos(x[:,1]) + s
                
        return f

class Ackley():
    def __init__(self, noise_var = 0, trace = False):
        self.domain = np.array([])

        self.params = {
            'a': 20,
            'b': 0.2,
            'c': 2 * torch.pi
        }

        self.trace = trace

    def evaluate(self, x):
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        n = x.shape[1]        
        first_operand = -a * torch.exp(torch.sqrt(torch.sum(x**2, axis = 1) / n) * -b)
        second_operand = torch.exp(torch.sum(np.cos(c * x), axis = 1) / n)

        if self.trace:
            print(f"t1:{torch.sqrt(torch.sum(x**2, axis = 1) / n)}")
            print(f"first operand: {first_operand}")
            print(f"second operand: {second_operand}")

        return first_operand - second_operand + a + torch.exp(torch.tensor(1))