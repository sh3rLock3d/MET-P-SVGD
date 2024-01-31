import torch
import numpy as np
from torch import autograd
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import pickle
import random
import os
import math
import shutil
alt.data_transformers.enable('default', max_rows=None)
from torch.optim import Adam
import torch.nn as nn
import time
import collections

#____________Hessian implementation
#TODO: 
#___python -m pip install git+https://github.com/mariogeiger/hessian.git 
# from hessian import hessian
#______________________--
from torch.autograd.functional import jacobian, hessian


from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)
# Global Variables
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class GMMDist(object):
    def __init__(self, dim, n_gmm, c_):
        self.dim = dim
        self.mix_probs = (1.0/n_gmm) * torch.ones(n_gmm)
        self.means = torch.tensor([[0.0, 0.0], [3.0, 2.0], [1.0, -0.5], [2.5, 1.5], [c_,c_]])
        self.std = torch.tensor([[0.16, 1.0], [1.0, 0.16], [0.5, 0.5], [0.5, 0.5], [0.5,0.5]])

    def sample(self, n):
        mix_idx = torch.multinomial(self.mix_probs, n[0], replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):    
        logps = []

        for i in range(len(self.mix_probs)):
            try:
                tmp = (- 0.5 * ( torch.matmul(torch.matmul((samples - self.means[i]).unsqueeze(2) , torch.diag(1/self.std[i]).unsqueeze(0).unsqueeze(1)) , (samples - self.means[i]).unsqueeze(-1) )) - 0.5 * np.log(  ((2 * np.pi)**self.dim) * self.std[i].prod() )) + self.mix_probs[i].log()
                logps.append(tmp.squeeze())
                #logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log( 2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
            except:                
                tmp = (- 0.5 * ( torch.matmul(torch.matmul((samples - self.means[i]).unsqueeze(1) , torch.diag(1/self.std[i]).unsqueeze(0)) , (samples - self.means[i]).unsqueeze(-1) )) - 0.5 * np.log(  ((2 * np.pi)**self.dim) * self.std[i].prod() )) + self.mix_probs[i].log()
                logps.append(tmp.squeeze())
        
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp

class Optim():
    """
    Optimizer Class 
    Inputs:
        lr: learning rate
    """
    def __init__(self, lr=None):
        self.m_dx, self.v_dx = 0, 0
        self.lr = lr
    
    def step(self,x, dx): 
        """
        update a set of particles in the direction of the gradient
        Inputs:
            x: a set of particles
            dx: the gradient vector
        Outputs:
            x: updated particles
        """
        dx = dx.view(x.size())
        x = x + self.lr * dx 
        
        
        #print('***',torch.abs(self.lr * dx).max())
        return x

class RBF:
    """
    Radial basis funtion kernel (https://en.wikipedia.org/wiki/Radial_basis_function)
    Inputs:
        sigma: Kernel standard deviatino 
        num_particles: number of particles
    """
    def __init__(self, sigma, num_particles):
        self.sigma = sigma
        self.num_particles = num_particles
        

    def forward(self, input_1, input_2):
        """
        Given two sets of points, return the matrix of distances between each point
        Inputs:
            input_1, input_2: a set of points
        Outputs:
            kappa: RBF matrix 
            diff: signed distance 
            h: kernel variance
            kappa_grad: derivative of rbf kernel
            gamma: 1/2*sigma**2
        """
        assert input_2.size()[-2:] == input_1.size()[-2:]
        
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        if self.sigma == "mean":
            median_sq = torch.mean(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)#[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
            gamma = 1.0 / (1e-8 + 2 * sigma**2) 
            
        elif self.sigma == "forth":
            median_sq = 0.5 * torch.mean(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)#[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
            gamma = 1.0 / (1e-8 + 2 * sigma**2) 
            
        elif self.sigma == "median":
            median_sq = torch.median(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
            gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        
        else:
            sigma = self.sigma
            gamma = 0.5 / (1e-8 + sigma**2) 
            h = None
        
        kappa = (-gamma * dist_sq).exp() 
        
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(), diff, h, kappa_grad, gamma
    
class Entropy_toy():
    
    def __init__(self, P, K, optimizer, num_particles, particles_dim, tb_logger):
        self.P = P
        
        self.optim = optimizer
        self.num_particles = num_particles
        self.particles_dim = particles_dim
        self.tb_logger = tb_logger
        self.K = K
       
        # svgd variables
        
        self.identity_mat = torch.eye(self.particles_dim).to(device)
        self.identity_mat2 = torch.eye(self.num_particles).to(device)
        
        # entropy varaibles
        self.logp_line1 = 0
        self.logp_line2 = 0
        self.logp_line3_A = 0
        self.logp_line3_B = 0 
        self.logp_line3_C = 0 
        
        # hessian terms
        
        self.hessian_of_log_prob = 0
        self.term_3_A = 0
        

    def SVGD(self,X):
        """
        Compute the Stein Variational Gradient given a set of particles
        Inputs:
            X: A set of particles
        Outputs:
            phi: the SVGD gradient
            phi_entropy: SVGD gradient with kernel without the distance of the particle to itself 
            (used to calculate the entropy)
        """

        
        
        #_____________________Compute the Hessian of log probability with torch.autograd.functional.hessian
        X.requires_grad_(True)
        
        self.get_log_prob = lambda X:self.P.log_prob(X)
        self.get_flat_log_prob = lambda X:self.P.log_prob(X).sum() 
       
        self.score_func = jacobian(self.get_log_prob, X).sum(1).reshape(self.num_particles, self.particles_dim)
        
        self.hessian_of_log_prob = hessian(self.get_flat_log_prob, X).sum(2)
        
        #https://pytorch.org/docs/stable/generated/torch.diagonal.html
        self.term_3 = torch.diagonal(self.hessian_of_log_prob, 0, -2, -1).sum(-1)
        
        
        
       
        self.K_XX, self.K_diff, self.K_h, self.K_grad, self.K_gamma = self.K.forward(X, X)  

        
        self.phi_term1 = self.K_XX.matmul(self.score_func) / self.num_particles
        self.phi_term2 = self.K_grad.sum(0) / self.num_particles
        
        phi = self.phi_term1 + self.phi_term2

        
        phi_entropy = phi
        

        return phi, phi_entropy
        
    def compute_logprob(self, phi, X):
        """
        Compute the log probability of the given particles in 3 ways 
        Inputs:
            X: A set of particles
            phi: The gradient used to update the particles 
        Outputs:
            self.logp_line1: log probability using line 1 in the presentation
            self.logp_line2: log probability using line 2 in the presentation
            self.logp_line3: log probability using line 3 in the presentation
        """
        grad_phi =[]

        for i in range(len(X)):
            grad_phi_tmp = []
            for j in range(self.particles_dim):
                grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach()
                grad_phi_tmp.append(grad_)
            grad_phi.append(torch.stack(grad_phi_tmp))
        
        
        self.grad_phi = torch.stack(grad_phi) 
        grad_phi_trace = torch.stack( [torch.trace(grad_phi[i]) for i in range(len(grad_phi))] ) 
        #_____ experiment
        #_____ instead of doing that for loop what if we just apply the jacobian haha
        self.grad_phi_experiment = jacobian(self.SVGD, X)[0].sum(2)
        grad_phi_ex_trace = torch.diagonal(self.grad_phi_experiment, 0, -2, -1).sum(-1)
        #_____ line 1
        
        self.logp_line1 = self.logp_line1 - torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr * self.grad_phi)))
        
        #_____ line 2
        
        self.logp_line2 = self.logp_line2 - self.optim.lr * grad_phi_ex_trace 
        
        
        #_____ line 3
        line3_term1 = (self.K_grad * self.score_func.unsqueeze(0)).sum(-1).sum(1)/(self.num_particles)
        line3_term2 = -2 * self.K_gamma * (( self.K_grad.permute(1,0,2) * self.K_diff).sum(-1) - self.particles_dim * (self.K_XX - self.identity_mat2) ).sum(0)/(self.num_particles)
        
        line3_term3 = self.term_3 /(self.num_particles)
        
        
        self.logp_line3 = self.logp_line3_A - self.optim.lr * (line3_term1 + line3_term2+ line3_term3) #replace line3_term3_b
        
        
        
        
            
    def step(self, X):
        """
        Perform one update step
        Inputs:
            X: A set of particles
            X: A set of updated particles
            phi_X: The gradient used to update the particles
        """
        
        phi_X, phi_X_entropy  = self.SVGD(X)  
        X_new = self.optim.step(X, phi_X) 
        self.compute_logprob(phi_X_entropy, X)
        X = X_new.detach() 
        
        return X, phi_X
    
        
def my_experiment(dim, num_particles, num_steps, kernel_sigma, lr, mu_init, sigma_init):
    
    mu = torch.full((dim,), mu_init).to(device).to(torch.float32)
    sigma = (torch.eye(dim) * sigma_init).to(device)
    init_dist = torch.distributions.MultivariateNormal(mu,covariance_matrix=sigma)
    X_init = init_dist.sample((num_particles,)).to(device)

    
    dist = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),covariance_matrix= 5 *torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))
    # dist = torch.distributions.MultivariateNormal(torch.zeros(3), 5 * torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

    experiment = Entropy_toy(dist, RBF(kernel_sigma, num_particles), Optim(lr), num_particles=num_particles, particles_dim=dim, tb_logger=tb_logger) 



    def main_loop(X, steps):
        """
        Perform SVGD updates
        Inputs:
            X: A set of particles
            steps: The number of SVGD steps
        Outputs:
            sampler_entr_svgd: the SVGD Entropy
            gt_entr: the ground truth Entropy
            charts: the set of density plots with particles.(a plot for each svgd step)
        """
        
        
        entr_values = []
        for t in range(steps): 
            print('__________________',t)
            
            X, phi_X = experiment.step(X)
           
            
            e1 = -(init_dist.log_prob(X_init) + experiment.logp_line1).mean().item()
            e2 = -(init_dist.log_prob(X_init) + experiment.logp_line2).mean().item()
            e3 = -(init_dist.log_prob(X_init) + experiment.logp_line3_A).mean().item()
            
            entr_values.append([e1, e2, e3])

            
        
        
        
        gt_entr = - dist.log_prob(dist.sample((500,))).mean()
        
        #sampler_entr =  -(init_dist.log_prob(X_init) + experiment.logp_line3).mean().item()
        sampler_entr =  -(init_dist.log_prob(X_init) + experiment.logp_line1).mean().item()
        
        return gt_entr, sampler_entr,  entr_values
    
    gt_entr, sampler_entr,  entr_values = main_loop(X_init.clone().detach(), steps=num_steps)
    
    return gt_entr, sampler_entr,  entr_values


dim = 2
# Learning Rate
lr = 0.5
# Number of mixtures
n_gmm = 1
# Target distribution's standard deviation, means are computed automatically
gmm_std = 1
# Number of particles
num_particles = 5
# Number of update steps
num_steps = 100
# Kernel variance
kernel_variance = 5

mu_init = 0
# Standard deviation of the initial distribution
sigma_init = 6
# dim, num_particles,  kernel_sigma, lr, mu_init, sigma_init, range_, tb_logger):

# tensorboard
if not os.path.exists("./exp"):os.makedirs("./exp") 

if not os.path.exists("./exp/figs"):os.makedirs("./exp/figs") 
#else: shutil.rmtree("./exp/figs")

if not os.path.exists("./tb_logs"):os.makedirs("./tb_logs")
#else: shutil.rmtree("./exp/tb_logs")
print('ddd')
tb_logger = SummaryWriter("./tb_logs/Implementing_Hessian" + "_"+datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))

times = {'hessian_by_func.hessian':[], 'hessian_by_loop':[], 'jacobian_of_score':[]}
gt_entr, sampler_entr,  entr_values = my_experiment(dim, num_particles, num_steps, kernel_variance, lr, mu_init, sigma_init)

ent = entr_values
ent1 = [ent[i][0] for i in range(len(entr_values))]
ent2 = [ent[i][1] for i in range(len(entr_values))]
ent3 = [ent[i][2] for i in range(len(entr_values))]


df = pd.DataFrame({'svgd_step': np.arange(len(entr_values)), 'line1': ent1, 'line2': ent2,'line3a': ent3})

# Create the initial plot
fig, ax1 = plt.subplots()

# Plot the entropy lines on the primary y-axis
ax1.axhline(y=gt_entr, color='r', linestyle='-', label='gt_entropy')
ax1.plot(df['svgd_step'], df['line1'], label='line 1', color='#006400')
ax1.plot(df['svgd_step'], df['line2'], label='line 2', color='#00CED1')
ax1.plot(df['svgd_step'], df['line3a'], label='line 3 heavy hess', color='#000080')

ax1.set_xlabel('svgd_steps')
ax1.set_ylabel('entropies', color='#006400')
ax1.set_xlim(0, len(entr_values))
plt.legend(['GT','entropy line1','entropy line2', 'heavy'])
plt.show()
 
 

print('hahahah')
print('end')