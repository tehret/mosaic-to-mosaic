import math
import torch as th
from torch.optim import Optimizer
import numpy as np

class SVAG(Optimizer):
    """Implements the SVAG algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Dissecting Adam\: The Sign, Magnitude and Variance of Stochastic Gradients
        https://arxiv.org/pdf/1705.07774.pdf
    """

    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(SVAG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SVAG, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SVAG does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = th.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = th.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta = group['beta']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta).add_(1 - beta, grad)
                exp_avg_sq.mul_(beta).addcmul_(1 - beta, grad, grad)

                beta_power = beta ** state['step']

                # initialization-bias correction
                bias_correction = 1.0 - beta_power
                m = exp_avg / bias_correction
                v = exp_avg_sq / bias_correction

                rho = (1.0 - beta) * (1.0 + beta_power)
                rho /= (1.0 + beta) * (1.0 - beta_power)
                rho = np.clip(rho, 0, 0.99999)  # avoid issue with rho at first iter

                # variance estimate
                m2 = m**2
                s = (v - m2) / (1.0 - rho)

                # adaptation factor
                gamma =  m2 / (m2 + s)

                # remove NaNs (if m = v = 0)
                gamma = th.clamp(gamma, 0.0, 1.0)

                step_size = group['lr'] 

                p.data.addcmul_(-step_size, grad, gamma)

        return loss

class MSVAG(Optimizer):
    """Implements the M-SVAG algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Dissecting Adam\: The Sign, Magnitude and Variance of Stochastic Gradients
        https://arxiv.org/pdf/1705.07774.pdf
    """

    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(MSVAG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MSVAG, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('MSVAG does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = th.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = th.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta = group['beta']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta).add_(1 - beta, grad)
                exp_avg_sq.mul_(beta).addcmul_(1 - beta, grad, grad)

                beta_power = beta ** state['step']

                # initialization-bias correction
                bias_correction = 1.0 - beta_power
                m = exp_avg / bias_correction
                v = exp_avg_sq / bias_correction

                rho = (1.0 - beta) * (1.0 + beta_power)
                rho /= (1.0 + beta) * (1.0 - beta_power)
                rho = np.clip(rho, 0, 0.99999)  # avoid issue with rho at first iter

                # variance estimate
                m2 = m**2
                s = (v - m2) / (1.0 - rho)

                # adaptation factor
                gamma =  m2 / (m2 + s.mul(rho))

                # remove NaNs (if m = v = 0)
                gamma = th.clamp(gamma, 0.0, 1.0)

                step_size = group['lr'] 

                p.data.addcmul_(-step_size, m, gamma)

        return loss
