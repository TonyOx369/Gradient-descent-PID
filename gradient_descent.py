import numpy as np

class GradientDescent:
    def __init__(self, learning_rate, a_min=None, a_max=None):
        self.learning_rate = learning_rate
        self.a_min = a_min
        self.a_max = a_max
        self.points = []
        self.result = []
        #self.G = np.zeros([len(a), len(a)])                  
        self.G = np.zeros([3, 3])                               # a contains kp ki kd
    
    def grad(self, a, cost_function_at_a, cost_function_a_h):
        h = 0.0000001
        grad = []
        for i in range(0, len(a)):
            grad.append((cost_function_a_h - cost_function_at_a) / h)
        grad = np.array(grad)
        return grad
    
    def update_a(self, learning_rate, grad):
        if len(grad) == 1:
            grad = grad[0]
        self.a -= (learning_rate * grad)
        if (self.a_min is not None) or (self.a_max is not None):
            self.a = np.clip(self.a, self.a_min, self.a_max)
    
    def update_G(self, grad):
        self.G += np.outer(grad,grad.T)
    
    def execute(self, a, cost_function, cost_function_a_h):
        self.a = a
        self.cost_function = cost_function
        self.cost_function_a_h = cost_function_a_h
        self.G = np.zeros([len(a), len(a)])
        self.points.append(list(self.a))
        self.result.append(self.cost_function)
        grad = self.grad(self.a, self.cost_function, self.cost_function_a_h)
        self.update_a(self.learning_rate, grad)
        return self.a
    
    def execute_adagrad(self, a, cost_function, cost_function_a_h):
        self.a = a
        self.cost_function = cost_function
        self.cost_function_a_h = cost_function_a_h
        self.points.append(list(self.a))
        self.result.append(self.cost_function)
        grad = self.grad(self.a, self.cost_function, self.cost_function_a_h)
        self.update_G(grad)
        self.learning_rate = self.learning_rate * np.diag(self.G)**(-0.5)
        self.update_a(self.learning_rate, grad)
        return self.a