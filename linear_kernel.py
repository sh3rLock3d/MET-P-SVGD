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
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
# Global Variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

torch.manual_seed(999)

# Helper Functions
def get_density_chart(P, d=7.0, step=0.1):
    """
    Given a probability distribution, return a density chart (Heatmap)
    Inputs:
        P: Probability distribution
        d: value used to bound the meshgrid
        step: value used in the arange method to create the meshgrid
    Outputs:
        chart: Altair object corresponding to a density plot
    """
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()

    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),})

    chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('p:Q', scale=alt.Scale(scheme='lightorange')),
    tooltip=['x','y','p']).properties(
    width=220,
    height=190
)

    return chart


def get_particles_chart(X, X_svgd=None):

    """
    Given a set of points, return a scatter plot
    Inputs:
        X: points (in our case. final position of the particles after applying svgd)
        X_svgd: intermidiate particles positions while applying svgd. if None do not add them to the plot
    Outputs:
        chart: Altair object corresponding to a scatter plot
    """
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],})

    chart = alt.Chart(df).mark_circle(color='black').encode(x='x:Q',y='y:Q')

    if X_svgd is not None:
        for i in range(np.shape(X_svgd)[1]):
            df_trajectory = pd.DataFrame({'x': X_svgd[:,i,0],'y': X_svgd[:,i,1],})
            chart += alt.Chart(df_trajectory).mark_line().mark_circle(color='green').encode(x='x:Q',y='y:Q')

    return chart

# Kernels
class Kernel1:
    """
    Radial basis funtion kernel (https://en.wikipedia.org/wiki/Radial_basis_function)
    Inputs:
        sigma: Kernel standard deviatino 
        num_particles: number of particles
    """
    def __init__(self, num_particles):
        self.num_particles = num_particles

    def forward(self, input_1, input_2):
        
        kappa = torch.matmul(input_1, input_2.t())
        grad = input_2        
        return kappa, grad



class Kernel2a:
    def __init__(self, num_particles, dim):
        self.num_particles = num_particles
        self.particles_dim = dim
        self.one_v = torch.ones((num_particles, dim)).to(device)
        self.norm_func = lambda x: torch.linalg.norm(x, axis = 1)
        self.kappa_func = lambda x, y: torch.matmul(x*(self.norm_func(x).pow(-1)).unsqueeze(1), (y*(self.norm_func(y).pow(-1)).unsqueeze(1)).t())
        self.identity = torch.eye(num_particles).to(device)
        

    def forward(self, input_1, input_2):
        kappa = (self.kappa_func(input_1, input_2)).to(device) 
        # 1/||X_i||
        V = self.norm_func(input_1).pow(-1).to(device)
        # 1/||X_i||^2
        V_sq = V.pow(2)
        
        # kappa_grad_1 = ((input_1*V.unsqueeze(1)).unsqueeze(1))*(V.view(-1, 1).expand(self.num_particles,self.num_particles).unsqueeze(2))
        kappa_grad_1 = torch.permute(torch.matmul(V.unsqueeze(0).unsqueeze(-1), torch.permute(input_1*V.unsqueeze(1), (-1,0)).unsqueeze(1)), (-1, -2, 0))
        kappa_grad_2 = -(kappa * V_sq).unsqueeze(-1)*input_1.unsqueeze(0)
         
        kappa_grad = (kappa_grad_1 + kappa_grad_2)

        return kappa, kappa_grad
    
#TODO: kernel where factor of normalization is a constant   
class Kernel2b:
    def __init__(self, num_particles, dim):
        self.num_particles = num_particles
        self.particles_dim = dim
        self.norm_func = lambda x: torch.linalg.norm(x, axis = 1)
        self.kappa_func = lambda x, y: torch.matmul(x*(self.norm_func(x).pow(-1)).unsqueeze(1), (y*(self.norm_func(y).pow(-1)).unsqueeze(1)).t())
        
    def forward(self, input_1, input_2):
        kappa = (self.kappa_func(input_1, input_2) + 1).to(device) 
        V = self.norm_func(input_1).pow(-1).to(device)
        indices = np.arange(0,self.num_particles)
        kappa_grad = torch.permute(torch.matmul(V.unsqueeze(0).unsqueeze(-1), torch.permute(input_1*V.unsqueeze(1), (-1,0)).unsqueeze(1)), (-1, -2, 0))
        kappa_grad[indices, indices] = 0
        # kappa_grad = ((input_1*V.unsqueeze(1)).unsqueeze(1))*(V.view(-1, 1).expand(self.num_particles,self.num_particles).unsqueeze(2))
        return kappa, kappa_grad

class RBF:
    """
    Radial basis funtion kernel (https://en.wikipedia.org/wiki/Radial_basis_function)
    Inputs:
        sigma: Kernel standard deviation
        num_particles: Number of particles
    """
    def __init__(self, sigma, num_particles):
        self.sigma = sigma
        self.num_particles = num_particles

    def forward(self, input_1, input_2):
        """
        Given two tensors of particles, return the matrix of distances between each particle
        Inputs:
            input_1: Particles coordinates
            input_2: Particles coordinates
        Outputs:
            kappa: RBF matrix
            diff: Signed distance
            h: Kernel variance
            kappa_grad: Derivative of rbf kernel
            gamma: 1/2*sigma**2
        """
        # Check if the inputs have the same size in the last two dimensions
        assert input_2.size()[-2:] == input_1.size()[-2:]

        # Compute the difference between each particle
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        # Square the difference and sum over the particle's dimensions
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)

        if self.sigma == "mean":
            # Estimate the kernel variance using the mean of all the particles distances
            median_sq = torch.mean(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
        elif self.sigma == "forth":
            # Estimate the kernel variance using the mean devided by two (one forth) of all the particles distances
            median_sq = 0.5 * torch.mean(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
        elif self.sigma == "median":
            # Estimate the kernel variance using the median of all the particles distances
            median_sq = torch.median(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
        else:
            # Setting the kernel variance from the input
            sigma = self.sigma
            h = None

        gamma = 1.0 / (1e-8 + 2 * sigma**2)

        kappa = (-gamma * dist_sq).exp()
        # Computing the gradient of the kernel over the particles
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(), kappa_grad


#Optimizer
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

        #print(torch.abs(self.lr * dx).max())
        return x


#Entropy Toy Class
class Entropy_toy():
    """
    Toy Class 
    Inputs:
        P: GMM Object
        K: RBF Kernel
        optimizer: Gradient Ascent optimizer
        num_particles: Number of particles
        particles_dim: particles number of dimensions 
        with_logprob: boolean decide if to compute the log probability or not
    """
    def __init__(self, P, K, optimizer, num_particles, particles_dim, with_logprob, num_svgd_steps, tb_logger, calculate_other_lines = True):
        self.P = P
        self.optim = optimizer
        self.num_particles = num_particles
        self.particles_dim = particles_dim
        self.num_svgd_steps = num_svgd_steps
        self.with_logprob = with_logprob
        self.K = K
        self.tb_logger = tb_logger
        
        # svgd variables
        self.identity_mat = torch.eye(self.particles_dim).to(device)
        self.identity_mat2 = torch.eye(self.num_particles).to(device)
        
        # entropy varaibles
        self.logp_line1 = 0
        self.logp_line2 = 0
        self.logp_line3 = 0

        # Kernel std loss 
        self.kernel_loss_Entr = torch.tensor(0.0).to(device)
        self.kernel_loss_SI = torch.tensor(0.0).to(device)

        self.get_log_prob = lambda x : self.P.log_prob(x).sum() #31/1
        self.calculate_other_lines = calculate_other_lines

                

    
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
        X = X.requires_grad_(True)
        
        #31/1
        #log_prob = self.P.log_prob(X)
        #score_func = autograd.grad(log_prob.sum(), X, create_graph=True)[0].reshape(self.num_particles, self.particles_dim)
        #self.score_func = score_func.reshape(self.num_particles, self.particles_dim)
        # self.score_func = jacobian(self.get_log_prob, X, create_graph=self.calculate_other_lines)
        self.score_func = jacobian(self.get_log_prob, X, create_graph=True)
        self.grad_score_func_trace = torch.vmap(torch.trace)(hessian(self.get_log_prob, X, create_graph=True)[np.arange(self.num_particles),:,np.arange(self.num_particles),:]) 

        self.K_XX, self.K_grad = self.K.forward(X, X)  

        self.num_particles =  self.num_particles
        self.phi_term1 = self.K_XX.matmul(self.score_func) / self.num_particles
        self.phi_term2 = self.K_grad.sum(0) / self.num_particles
        phi = self.phi_term1 + self.phi_term2
        
        phi_entropy = (self.K_XX-self.identity_mat2).matmul(self.score_func) / (self.num_particles-1)
        phi_entropy += (self.K_grad.sum(0) / (self.num_particles-1))
        
        31/1
        # phi_entropy = phi
        
        return phi, phi_entropy
    
    
    def compute_stein_identity(self, X):
        #phi [200,2]
        #score [200,2]
        X = X.requires_grad_(True)    
        phi, phi_X_entr = self.SVGD(X)
        
        phi = phi_X_entr
        
        grad_phi =[]
        for i in range(len(X)):
            grad_phi_tmp = []
            for j in range(self.particles_dim):
                grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach() #safa
                grad_phi_tmp.append(grad_)
            grad_phi.append(torch.stack(grad_phi_tmp))

        grad_phi = torch.stack(grad_phi) 
        SI = phi.unsqueeze(-1).matmul(self.score_func.unsqueeze(-2))
        SI += grad_phi 
        SId = SI.mean(0).sum()
        SD = (torch.stack([SI[i].diag().sum() for i in range(len(SI)) ]).mean())**2
        #TODO: Visualize KLD, SId, SImax, SImin
        SImax, SImin = torch.max(SI), torch.min(SI)
        return SId, SD, SImax, SImin
    
    
    def compute_logprob(self, phi, X, svgd_itr):
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
        if self.calculate_other_lines:
                grad_phi =[]
                for i in range(len(X)):
                    grad_phi_tmp = []
                    for j in range(self.particles_dim):
                        grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach() #safa
                        grad_phi_tmp.append(grad_)
                    grad_phi.append(torch.stack(grad_phi_tmp))

                self.grad_phi = torch.stack(grad_phi) 
                
                # det_mat = torch.det(self.identity_mat + self.optim.lr * self.grad_phi)
                #self.tb_logger.add_scalars('det_Jaccobian ', {'min ': det_mat.min().item(), 'max ':det_mat.max().item(),  'mean ':det_mat.mean().item()   } , svgd_itr)
                #self.tb_logger.add_histogram( 'det_Jaccobian ', det_mat, svgd_itr )
                
                self.logp_line1 = self.logp_line1 - torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr * self.grad_phi)))

                # grad_phi_trace = torch.stack( [torch.trace(grad_phi[i]) for i in range(len(grad_phi))] ) 
                # self.logp_line2 = self.logp_line2 - self.optim.lr * grad_phi_trace
        '''
        #_________________ we don't want to visualize line 3atm
        line3_term1 = (2.0*self.K_grad * self.score_func.unsqueeze(0)).sum(-1).squeeze()/(self.num_particles)
        #line3_term1 = (self.K_grad * self.score_func.unsqueeze(0)).sum(-1).sum(1)/(self.num_particles)

        line3_term2 = 2 * torch.ones(self.num_particles)/(self.num_particles)
        #line3_term2 = torch.ones(self.num_particles)/(self.num_particles)

        line3_term3 = self.grad_score_func_trace / (self.num_particles) #31/1

        invertability = line3_term1 + line3_term2 + line3_term3 #31/1
        
        self.logp_line3 = self.logp_line3 - self.optim.lr * (line3_term1 + line3_term2 + line3_term3) #31/1 
        '''
        '''
        if (svgd_itr > int(self.sigma_k_param_loss_step*self.num_svgd_steps)): #new # loss of 10 percent last steps
            if (self.sigma_k_param_loss=="Entr"):
                self.kernel_loss_Entr += (invertability).mean(0)
            elif (self.sigma_k_param_loss=="Entr_sq"):
                self.kernel_loss_Entr += ((invertability)**2).mean(0)
            elif (self.sigma_k_param_loss=="Entr_sq_weighted"):
                self.kernel_loss_Entr += ((svgd_itr/self.num_svgd_steps)*((invertability)**2)).mean(0)
            elif (self.sigma_k_param_loss=="Entr_l1_weighted"):
                self.kernel_loss_Entr += ((svgd_itr/self.num_svgd_steps)*(torch.abs(invertability))).mean(0)
        '''

        
        
        
    def step(self, X, V=None, alg=None, svgd_itr=None):
        """
        Perform one update step
        Inputs:
            X: A set of particles
            alg: The name of the algorithm that should be used
        Outputs:
            X: A set of updated particles
            phi_X: The gradient used to update the particles
        """
        phi_X, phi_X_entropy = self.SVGD(X) 

        '''
        if (svgd_itr > int(self.sigma_k_param_loss_step*self.num_svgd_steps)):
            if (self.sigma_k_param_loss=="SI"): #new 
                #loss_SI_step = (svgd_itr/self.num_svgd_steps)* (phi_X).sum(-1).mean(0)
                loss_SI_step = (phi_X).sum(-1).mean(0)
                #loss_SI_step = ((phi_X).sum(-1)**2).mean(0)
                self.kernel_loss_SI += loss_SI_step
            if (self.sigma_k_param_loss=="SI_sq"):
                #loss_SI_step = (svgd_itr/self.num_svgd_steps)* (phi_X**2).sum(-1).mean(0)
                loss_SI_step =  (phi_X.sum(-1).mean(0))**2
                self.kernel_loss_SI += loss_SI_step
        
        self.tb_logger.add_scalar('loss_SI_step',  (phi_X).sum(-1).mean(0), svgd_itr)
        '''
        X_new = self.optim.step(X, phi_X) 
        
        if self.with_logprob: 
            self.compute_logprob(phi_X_entropy, X, svgd_itr)
        
        # check convergence
        p_dist = ((X_new-X)**2).sum(-1)        
        self.tb_logger.add_histogram( 'convergence ', p_dist, svgd_itr )
        
        X = X_new#.detach() #safa #new
        
        return X, phi_X 




def my_experiment(dim, n_gmm, num_particles, num_svgd_step, kernel_sigma, gmm_std, lr, mu_init, sigma_init, tb_logger,Project_name,plot, calculate_other_lines = False, kernel='2a'):
    """
    Perform one whole experiment 
    Inputs:
        dim: Number of dimensions of the particles
        n_gmm: Number of mixutres of the target GMM distribution
        num_particles: number of particles
        num_svgd_step: number of SVGD steps
        kernel_sigma: Kernel standard diviation
        gmm_std: The target distribution standard diviation
        lr: learning rate
        sigma_init: standard deviation of the initial distribution
        mu_init: mean of the initial distribution
    Outputs:
        sampler_entr_svgd: the SVGD Entropy
        sampler_entr_ld: The LD Entropy
        gt_entr: the ground truth Entropy
        charts: the set of density plots with particles.(a plot for each svgd step)
    """
    mu = torch.full((dim,), mu_init).to(device).to(torch.float32)
    sigma = (torch.eye(dim) * sigma_init).to(device)
    init_dist = torch.distributions.MultivariateNormal(mu,covariance_matrix=sigma)
    X_init = init_dist.sample((num_particles,)).to(device)
    
    ###########################
    #     gauss = torch.distributions.MultivariateNormal(torch.Tensor([0.0,0.5]).to(device),covariance_matrix= torch.Tensor([[0.1,0.0],[0.0,0.1]]).to(device))
    gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),covariance_matrix= 5 *torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))
    #gauss = torch.distributions.exponential.Exponential(torch.tensor([5.0,5.0]))
    #gauss = torch.distributions.uniform.Uniform(torch.tensor([-7.0,-7.0]), torch.tensor([7.0,7.0]))
    #     gauss = GMMDist(dim=dim, n_gmm=n_gmm, sigma=gmm_std)
    if kernel == '2a':
        experiment = Entropy_toy(gauss, Kernel2a(num_particles, dim), Optim(lr), num_particles=num_particles, particles_dim=dim, with_logprob=True, num_svgd_steps=num_svgd_step, tb_logger=tb_logger, calculate_other_lines = calculate_other_lines) 
    else:
        experiment = Entropy_toy(gauss, Kernel2b(num_particles, dim), Optim(lr), num_particles=num_particles, particles_dim=dim, with_logprob=True, num_svgd_steps=num_svgd_step, tb_logger=tb_logger, calculate_other_lines = calculate_other_lines) 
    if dim == 2:
        gauss_chart = get_density_chart(gauss, d=7.0, step=0.1) 
        init_chart = gauss_chart + get_particles_chart(X_init.cpu().numpy())
    
    ####verify if kernel is in the Stein Class
    #sample from target distribution
    target_samples = gauss.sample((num_particles,))
    #target_phi_X, target_phi_X_entr = experiment.SVGD(target_samples)
    #tb_logger.add_histogram( 'Stein Identity/phi ', (target_phi_X**2).sum(-1) , 0 )
    #tb_logger.add_histogram( 'Stein Identity/phi_entr ', (target_phi_X_entr**2).sum(-1), 0 )
    SI, SD, SImax, SImin = experiment.compute_stein_identity(target_samples)
    
    def main_loop(alg, X, steps):
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
        gt_entr = - gauss.log_prob(gauss.sample((500,))).mean()
        charts = []
        X_svgd_=[]
        ent = []
        conv = []
        # for t in tqdm(range(steps), desc='svgd_steps'): 
        for t in range(steps): 
            print('__________________',t)
            X, phi_X = experiment.step(X, alg,  svgd_itr=t)
            if plot:
                X_svgd_.append(X.clone().detach().clone())

            if plot and (dim == 2):
                chart_ = gauss_chart + get_particles_chart(X.clone().detach().cpu().numpy())
                charts.append(chart_)
                
            
                
            ent1 = -(init_dist.log_prob(X_init) + experiment.logp_line1).mean().item()
            tb_logger.add_scalars('SVGD_Entr',{'entropy_1':ent1, 'gt_entropy':gt_entr}, t)
            #______________________ KL divergence
            KLD = -(experiment.P.log_prob(X).mean() + experiment.logp_line1)
            tb_logger.add_scalar('KL_divergence', KLD.mean(), t)
            
            # ent1 = -(init_dist.log_prob(X_init) + experiment.logp_line1).mean().item()
            # ent2 = -(init_dist.log_prob(X_init) + experiment.logp_line2).mean().item()
            
            # ent.append( [ent1, ent2, ent3])
            SI, SD, SImax, SImin = experiment.compute_stein_identity(target_samples)
            tb_logger.add_scalars('stein_parameters',{'Stein_identity':SI, 'SteinMax':SImax, 'SteinMin':SImin},t)
            conv.append(SI.item())
            '''
            if (sigma_k_param["train_1_sigma"]==False):
                tb_logger.add_scalar('SVGD_Sig_k', experiment.K.sigma[t], t)
            '''
        if plot:
            X_svgd_ = torch.stack(X_svgd_)
            chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
       
        
        
        sampler_entr =  -(init_dist.log_prob(X_init) + experiment.logp_line1).mean().item()                
        
        return gt_entr, sampler_entr, charts, X, ent, conv

    gt_entr, sampler_entr_svgd, charts_svgd, X, ent, conv = main_loop('svgd', X_init.clone(), steps=num_svgd_step)
    
    # loss_SI =  experiment.kernel_loss_SI 
    # loss_entr = experiment.kernel_loss_Entr
    
    if plot:
        (charts_svgd[0]|charts_svgd[20]|charts_svgd[40]|charts_svgd[60]|charts_svgd[80]|charts_svgd[-1]).save('./exp/figs/t'+str(kernel)+'_'+Project_name+'.html')
    
    print('____________________________________________________________')
    return init_chart, sampler_entr_svgd, gt_entr, charts_svgd, ent, conv

# tensorboard
if not os.path.exists("./exp"):os.makedirs("./exp") 

if not os.path.exists("./exp/figs"):os.makedirs("./exp/figs") 
#else: shutil.rmtree("./exp/figs")

if not os.path.exists("./tb_logs"):os.makedirs("./tb_logs")
#else: shutil.rmtree("./exp/tb_logs")




####
dim = 2
n_gmm = 1
gmm_std = 1
sigma_init = 6
mu_init = 0

# svgd paramters
lr = 0.07
num_particles = 100

num_steps = 400
kernel_sigma = 5

# learning of the kernel variance
sigma_k_param = {}
calculate_other_lines = True

# Testing different kernels
kernels = ['2a', '2b']

for kernel in kernels:
    # project name and logger 
    Project_name = "Kernel___"+str(kernel)
    Project_name += "_SVGD_lr "+str(lr)
    Project_name += "_num_particles_" +str(num_particles)
    Project_name += "_num_steps_"+str(num_steps)
    Project_name += "_"+datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    print(Project_name)
    tb_logger = SummaryWriter("./tb_logs/"+Project_name)
# plot
    plot=True

    print("*************************************")
    print(Project_name)
    print("*************************************")

    init_chart, sampler_entr_svgd, gt_entr, charts_svgd, ent, conv = my_experiment(dim=dim, n_gmm=n_gmm, num_particles=num_particles, num_svgd_step=num_steps, kernel_sigma=kernel_sigma, gmm_std=gmm_std, lr=lr, mu_init=mu_init, sigma_init=sigma_init, tb_logger=tb_logger,Project_name=Project_name,plot=plot, calculate_other_lines=calculate_other_lines, kernel = kernel)
    with open('/home/local/QCRI/elaabouazza/svgd/MET-P-SVGD/charts__' + str(kernel) + '.pkl','wb') as handler:
        pickle.dump(charts_svgd, handler)
    tb_logger.close()


'''
ent1 = [i[0] for i in ent]
ent2 = [i[1] for i in ent]
ent3 = [i[2] for i in ent]
df = pd.DataFrame({'svgd_step':np.arange(num_steps), 'line3':ent3, 'line2':ent2, 'line1':ent1})
plt.axhline(y = gt_entr.item(), color='r', linestyle='-', label = 'gt_entropy')
plt.plot(df['svgd_step'], df['line3'], label = 'line 3', color = '#006400')
plt.plot(df['svgd_step'], df['line2'], label = 'line 2', )
plt.plot(df['svgd_step'], df['line1'], label = 'line 1', )
plt.legend()
plt.xlabel('itr')
plt.ylabel('entropies')
plt.savefig('notrain.png')
plt.show()
'''
#init_chart.save('./init_chart.png')