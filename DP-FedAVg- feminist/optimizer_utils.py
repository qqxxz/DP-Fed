import torch
import torch.optim as optim
from opacus.optimizers import DPOptimizer

class ServerOptimizer:
    """Base class for server optimizers."""
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, model, grads):
        raise NotImplementedError

class SGDServerOptimizer(ServerOptimizer):
    """SGD optimizer for server."""
    def step(self, model, grads):
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
                param.data -= self.learning_rate * grad

class DPSGDMServerOptimizer(ServerOptimizer):
    """DP-SGD with momentum optimizer for server."""
    def __init__(self, learning_rate, momentum=0.9, noise_std=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.noise_std = noise_std
        self.velocity = None

    def step(self, model, grads):
        if self.velocity is None:
            self.velocity = [torch.zeros_like(p) for p in model.parameters()]
        
        with torch.no_grad():
            for i, (param, grad) in enumerate(zip(model.parameters(), grads)):
                # Add noise for differential privacy
                if self.noise_std > 0:
                    grad = grad + torch.randn_like(grad) * self.noise_std
                
                # Update velocity
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                
                # Update parameters
                param.data -= self.learning_rate * self.velocity[i]

class DPFTRLMServerOptimizer(ServerOptimizer):
    """DP-FTRL with momentum optimizer for server."""
    def __init__(self, learning_rate, momentum=0.9, noise_std=0.0, decay_rate=0.5):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.noise_std = noise_std
        self.decay_rate = decay_rate
        self.velocity = None
        self.squared_grads = None
        self.step_count = 0

    def step(self, model, grads):
        if self.velocity is None:
            self.velocity = [torch.zeros_like(p) for p in model.parameters()]
        if self.squared_grads is None:
            self.squared_grads = [torch.ones_like(p) for p in model.parameters()]
        
        self.step_count += 1
        
        with torch.no_grad():
            for i, (param, grad) in enumerate(zip(model.parameters(), grads)):
                # Add noise for differential privacy
                if self.noise_std > 0:
                    grad = grad + torch.randn_like(grad) * self.noise_std
                
                # Update squared gradients
                self.squared_grads[i] = (
                    self.decay_rate * self.squared_grads[i] + 
                    (1 - self.decay_rate) * grad ** 2
                )
                
                # Compute adaptive learning rate
                lr = self.learning_rate / (torch.sqrt(self.squared_grads[i]) + 1e-8)
                
                # Update velocity
                self.velocity[i] = self.momentum * self.velocity[i] + lr * grad
                
                # Update parameters
                param.data -= self.velocity[i]

def create_client_optimizer(model, optimizer_name, learning_rate):
    """Create client optimizer."""
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Unknown client optimizer name {optimizer_name}')

def create_dp_optimizer(model, optimizer, noise_multiplier, max_grad_norm):
    """Create differentially private optimizer."""
    return DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=1
    ) 