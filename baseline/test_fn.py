import torch
import numpy as np
import rover 
from rover_utils import ConstantOffsetFn, NormalizedInputFn

# Test functions available at https://github.com/zi-w/Ensemble-Bayesian-Optimization


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

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != 2:
            raise(Exception("Wrong function input dimension."))
        
        f = a*(x[:,1] - b*x[:,1]**2 + c*x[:,0] - r)**2 + s*(1-t)*torch.cos(x[:,1]) + s
                
        return f

class Ackley():
    def __init__(self, noise_var = 0, trace = False):
        self.domain = np.array([-32.768, 32.768])

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

        x = torch.clip(x, self.domain[0], self.domain[1])

        n = x.shape[1]        
        first_operand = -a * torch.exp(torch.sqrt(torch.sum(x**2, axis = 1) / n) * -b)
        second_operand = torch.exp(torch.sum(np.cos(c * x), axis = 1) / n)

        if self.trace:
            print(f"t1:{torch.sqrt(torch.sum(x**2, axis = 1) / n)}")
            print(f"first operand: {first_operand}")
            print(f"second operand: {second_operand}")

        return first_operand - second_operand + a + torch.exp(torch.tensor(1))

class Rover(): 
    def __init__(self): 
        domain = rover.create_large_domain()
        n_points = domain.traj.npoints
        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)

        f_max = 5.0
        f = ConstantOffsetFn(domain, f_max)
        self.fn = NormalizedInputFn(f, raw_x_range)

    
    def l2cost(self, x, point):
        return 10 * np.linalg.norm(x - point, 1)

    def evaluate(self, x): 
        if type(x) == type(torch.tensor): 
            x = x.detach().numpy()
        
        return torch.tensor(self.fn(x))

class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        
    def evaluate(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val
