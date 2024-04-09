import torch
import numpy as np
from torch import autograd
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import random
import math


alt.data_transformers.enable('default', max_rows=None)

#Global Variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Helper Functions
def get_density_chart(P, d=4.0, step=0.1):
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
    height=190)

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

#Kernels
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
        elif self.sigma == "forth":
            median_sq = 0.5 * torch.mean(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)#[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
        elif self.sigma == "median":
            median_sq = torch.median(dist_sq.detach().reshape(-1, self.num_particles*self.num_particles), dim=1)[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(self.num_particles + 1.))
            sigma = torch.sqrt(h)
        else:
            sigma = self.sigma
            h = None
        
        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        
        kappa = (-gamma * dist_sq).exp() 
        
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(), diff, h, kappa_grad, gamma

#GMM
class GMMDist(object):
    def __init__(self, dim, n_gmm, c_):
        self.dim = dim
        self.mix_probs = ((1.0/n_gmm) * torch.ones(n_gmm)).to(device)
        self.means = torch.tensor([[0.0, 0.0], [3.0, 2.0], [1.0, -0.5], [2.5, 1.5], [c_,c_]]).to(device)
        self.std = torch.tensor([[0.16, 1.0], [1.0, 0.16], [0.5, 0.5], [0.5, 0.5], [0.5,0.5]]).to(device)

    def sample(self, n):
        mix_idx = torch.multinomial(self.mix_probs, n[0], replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):    
        logps = []

        for i in range(len(self.mix_probs)):
            try:
                tmp = (- 0.5 * ( torch.matmul(torch.matmul((samples - self.means[i]).unsqueeze(2) , torch.diag(1/self.std[i]).unsqueeze(0).unsqueeze(1)) , (samples - self.means[i]).unsqueeze(-1) )) - 0.5 * (  ((2 * np.pi)**self.dim) * self.std[i].prod() ).log()) + self.mix_probs[i].log()
                logps.append(tmp.squeeze())
                #logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log( 2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
            except:                
                tmp = (- 0.5 * ( torch.matmul(torch.matmul((samples - self.means[i]).unsqueeze(1) , torch.diag(1/self.std[i]).unsqueeze(0)) , (samples - self.means[i]).unsqueeze(-1) )) - 0.5 * (  ((2 * np.pi)**self.dim) * self.std[i].prod() ).log()) + self.mix_probs[i].log()
                logps.append(tmp.squeeze())
        
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp



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
        
        viloation = int(torch.abs(self.lr * dx).max()>=1)
        # print('***',torch.abs(self.lr * dx).max())
        return x, viloation

#Entropy Toy Class
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
    def __init__(self, P, K, optimizer, num_particles, particles_dim, with_logprob, tb_logger):
        self.P = P
        self.optim = optimizer
        self.num_particles = num_particles
        self.particles_dim = particles_dim
        self.with_logprob = with_logprob
        self.K = K
        self.tb_logger = tb_logger

        # svgd variables
        mu_ld_noise = torch.zeros((self.particles_dim,)).to(device) 
        self.identity_mat = torch.eye(self.particles_dim).to(device)
        self.identity_mat2 = torch.eye(self.num_particles).to(device)
        
        # entropy varaibles
        self.logp_line1 = 0
        self.logp_line2 = 0
        self.logp_line3 = 0

    
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
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0].reshape(self.num_particles, self.particles_dim)
        self.score_func = score_func.reshape(self.num_particles, self.particles_dim)
        # print('######### score_func', (self.score_func).mean().detach().item())
        
        self.K_XX, self.K_diff, self.K_h, self.K_grad, self.K_gamma = self.K.forward(X, X)  

        self.num_particles =  self.num_particles
        self.phi_term1 = self.K_XX.matmul(score_func) / self.num_particles
        self.phi_term2 = self.K_grad.sum(0) / self.num_particles
        phi = self.phi_term1 + self.phi_term2
        
        phi_entropy = (self.K_XX-self.identity_mat2).matmul(score_func) / (self.num_particles-1)
        phi_entropy += (self.K_grad.sum(0) / (self.num_particles-1))
        
        #print((torch.norm(phi_entropy, dim=1).view(-1,1) ))

        #phi_entropy = phi_entropy * (torch.norm(phi_entropy, dim=1).view(-1,1) > 0.001).int() 

        return phi, phi_entropy


    
    
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
        grad_phi =[]
        for i in range(len(X)):
            grad_phi_tmp = []
            for j in range(self.particles_dim):
                grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach()
                grad_phi_tmp.append(grad_)
            grad_phi.append(torch.stack(grad_phi_tmp))
        
        #print(grad_phi)
        self.grad_phi = torch.stack(grad_phi) 
        self.logp_line1 = self.logp_line1 - torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr * self.grad_phi)))

        self.grad_phi_trace = torch.stack( [torch.trace(grad_phi[i]) for i in range(len(grad_phi))] ) 
        grad_phi_trace = self.grad_phi_trace
        self.logp_line2 = self.logp_line2 - self.optim.lr * grad_phi_trace
        #_____________________________implicit function theorem or det(jacobian)____________
        det_mat = torch.det(self.identity_mat + self.optim.lr * self.grad_phi)
        self.tb_logger.add_scalars('det_Jaccobian ', {'min ': det_mat.min().item(), 'max ':det_mat.max().item(),  'mean ':det_mat.mean().item()   } , svgd_itr)
        self.tb_logger.add_histogram( 'det_Jaccobian ', det_mat, svgd_itr )
        
        '''
        line3_term1 = (self.K_grad * self.score_func.unsqueeze(0)).sum(-1).sum(1)/(self.num_particles-1)
        line3_term2 = -2 * self.K_gamma * (( self.K_grad.permute(1,0,2) * self.K_diff).sum(-1) - self.particles_dim * (self.K_XX - self.identity_mat2) ).sum(0)/(self.num_particles-1)
        self.logp_line3 = self.logp_line3 - self.optim.lr * (line3_term1 + line3_term2)
        '''

    def step(self, X, V=None, alg=None, itr=None):
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
            
        X_new, violation = self.optim.step(X, phi_X) 
        
        if self.with_logprob: 
            self.compute_logprob(phi_X_entropy, X, svgd_itr=itr)
        
        # print('****************************convergence',((X_new-X)**2).sum(-1).mean())
        convergence = ((X_new-X)**2).sum(-1).mean()

        if (alg == 'svgd'):
            X = X_new.detach() 
        else:
            X = X_new
        
        return X, phi_X, violation, convergence 


#Main Loop
def my_experiment(dim, n_gmm, num_particles, num_svgd_step, kernel_sigma, gmm_std, lr, mu_init, sigma_init, sigma_V_init, c_, tb_logger):
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
     
    init_V_dist = torch.distributions.MultivariateNormal(torch.zeros(dim).to(device),covariance_matrix = (torch.eye(dim) * sigma_V_init).to(device))
    V_init = init_V_dist.sample((num_particles,)).to(device)
    ###########################

    #     gauss = torch.distributions.MultivariateNormal(torch.Tensor([0.0,0.5]).to(device),covariance_matrix= torch.Tensor([[0.1,0.0],[0.0,0.1]]).to(device))
    #gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),covariance_matrix= 5 *torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))
    #gauss = torch.distributions.exponential.Exponential(torch.tensor([5.0,5.0]))
    #gauss = torch.distributions.uniform.Uniform(torch.tensor([-7.0,-7.0]), torch.tensor([7.0,7.0]))
    gauss = GMMDist(dim=dim, n_gmm=n_gmm, c_=c_)

    experiment = Entropy_toy(gauss, RBF(kernel_sigma, num_particles), Optim(lr), num_particles=num_particles, particles_dim=dim, with_logprob=True, tb_logger = tb_logger) 

    if dim == 2:
        gauss_chart = get_density_chart(gauss, d=4.0, step=0.1) 
        init_chart = gauss_chart + get_particles_chart(X_init.cpu().numpy())
    

    def main_loop(alg, X, V, steps):
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
        charts = []
        X_svgd_=[]

        entr_values = []
        kl_values = []
        trace_h = []
        
        num_violation = 0
        gt_entr = - gauss.log_prob(gauss.sample((500,))).mean()
        
        for t in tqdm(range(steps), desc='svgd_steps'): 
        # for t in range(steps): 
            #print('__________________',t)
            #print('X_svgd_ ', X_svgd_)
            X, phi_X, violation, convergence = experiment.step(X, V, alg, t)
            num_violation += violation
            X_svgd_.append(X.detach().clone())
            
            # if convergence<0.00001:
            #     break
            
            
#             print(X.min(), X.mean(), X.max())
#             if t> 0 :
#                 print(X_svgd_[t-1] == X_svgd_[t])

            if dim == 2:
                chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
                # chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), torch.stack(X_svgd_).detach().cpu().numpy())
                charts.append(chart_)
            
            entropy_l_1 =  -(init_dist.log_prob(X_init) + experiment.logp_line1).mean().item()
            entropy_l_2 =  -(init_dist.log_prob(X_init) + experiment.logp_line2).mean().item()
            tb_logger.add_scalars('SVGD_Entr',{'entropy_1':entropy_l_1, 'entropy_2':entropy_l_2, 'gt_entropy':gt_entr}, t)
            #KL divergence visualization
            KLD = -(experiment.P.log_prob(X).mean() + experiment.logp_line1)
            
            kl_values.append(KLD.mean().item())
            trace_h.append(experiment.grad_phi_trace.mean().item())
            tb_logger.add_scalar('KL_divergence', KLD.mean(), t)
            entr_values.append(entropy_l_1)

            # print(t, ' entropy svgd (line 2): ',  -(init_dist.log_prob(X_init) + experiment.logp_line2).mean().item())
#             print(t, ' entropy svgd (line 3): ',  -(init_dist.log_prob(X_init) + experiment.logp_line3).mean().item())
            # print()
        X_svgd_ = torch.stack(X_svgd_)
        # chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
        
        
        
        sampler_entr =  -(init_dist.log_prob(X_init) + experiment.logp_line3).mean().item()
        
        
        return gt_entr, sampler_entr, charts, entr_values, num_violation, kl_values, trace_h

    # Run SVGD
    print('_________SVGD___________')
    gt_entr, sampler_entr_svgd, charts_svgd, entr_values_svgd, num_violation_svgd, kl_values, trace_h = main_loop('svgd', X_init.clone(), V_init.clone(), steps=num_svgd_step)
    
    print('____________________________________________________________')
    samplers = [sampler_entr_svgd, gt_entr]
    charts = [charts_svgd]
    entr_values = [entr_values_svgd]

    return init_chart, samplers, charts, entr_values, kl_values, trace_h



dim = 2
learning_rates = [0.5]
n_gmm = 5
num_steps = 800
gmm_std = 1.0
sigma_V_init = 1

c_list = [-3,-2,-1,0,1,2,3]


num_particles = 200
kernel_sigma = 7.0

mu_init=[0.0]
sigma_init=[3.0]


list_init_distr = list(zip(mu_init,sigma_init))


##
results = []
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

Project_name = "____Tuesday___"+str(kernel_sigma)
Project_name += "_num_particles_" +str(num_particles)
Project_name += "_num_steps_"+str(num_steps)
lr_add = ""

for lr in tqdm(learning_rates, desc='learning_rate'):
    # project name and logger 
    lr_add = "_SVGD_lr "+str(lr)
    


    for mu_init, sigma_init in tqdm(list_init_distr, desc='distribution_init'):
        
        traj=[]
        results = []
        final_charts = []
        for c_ in c_list:
            add = f"{mu_init}&{sigma_init}&c_={c_}"
            
            print(Project_name+ lr_add + add)
            tb_logger = SummaryWriter("./tb_logs/APRIL/"+Project_name + lr_add + add)
            init_chart, samplers, charts, entr_values, kl_values, trace_h = my_experiment(dim=dim, n_gmm=n_gmm, num_particles=num_particles, num_svgd_step=num_steps, kernel_sigma=kernel_sigma, gmm_std=gmm_std, lr=lr, mu_init=mu_init, sigma_init=sigma_init, sigma_V_init=sigma_V_init, c_=c_, tb_logger = tb_logger)
            
            df= pd.DataFrame({'itr': np.arange(len(entr_values[0])) ,'entropy': entr_values[0],'kld': kl_values,'trace_grad_h': trace_h})
            
            # Assuming 'df' is your original DataFrame based on the second image you uploaded.
            df_long = df.melt(id_vars='itr', value_vars=['entropy', 'kld', 'trace_grad_h'], var_name='type', value_name='value')

            # Now, create the Altair chart with the reshaped DataFrame.
            chart = alt.Chart(df_long).mark_line().encode(
                x='itr:Q',
                y='value:Q',
                color='type:N',
                tooltip=['itr', 'value', 'type']
            ).properties(
                title='Entropy & KLD & Trace_gradient_H'
            )

            
            # chart = alt.Chart(df_entr).mark_line().encode(x='itr',y='entropy').properties(title=f' entropy for c={c_}&\sigma={kernel_sigma}&n={num_particles}&sigma_init{sigma_init}')
            # kl_chart = alt.Chart(df_kl).mark_line().encode(x='itr',y='kld').properties(title=f'kldivergence for c={c_}&\sigma={kernel_sigma}&n={num_particles}&sigma_init{sigma_init}')
            # tr_chart = alt.Chart(df_tr).mark_line().encode(x='itr',y='trace_grad_h').properties(title=f'trace grad h for c={c_}&\sigma={kernel_sigma}&n={num_particles}&sigma_init{sigma_init}')
            # # (chart|charts[0][0]|charts[0][-1]).save('./charts/c_'+str(c_)+'kernel_'+str(kernel_sigma)+'_num_particles'+str(num_particles)+'_mu_init'+str(mu_init)+'sigma_init'+str(sigma_init)+'.html')
            
            charts_ = (chart|charts[0][0]|charts[0][-1]).properties(title='entropy plots + qualitative convergence')
            charts_.save('./charts/APRIL/'+'TUESDAY_04_09_kernel__'+'chart for c_=='+str(c_)+'.html')
            # tr_chart.save('./charts/APRIL/'+'TUESDAY_04_09_kernel__'+'trace gradient of h'+str(c_)+'.html')
            print(f'saved chart for c_={c_}')
            final_charts.append(charts_)
            traj.append(entr_values[0][-1])

            print('done')

        results.append([traj, [kernel_sigma,num_particles,mu_init,sigma_init]  ])
        alt.vconcat(*final_charts).save('./charts/APRIL/'+'TUESDAY_04_09_kernel__'+str(kernel_sigma)+'__num_particles__'+str(num_particles)+'_mu_init'+str(mu_init)+'sigma_init'+str(sigma_init)+'lr='+str(lr)+'.html')
        ############################################################################
        print('saved chart! Check it out!')
        # save results
        if kernel_sigma in ['forth', 'median', 'mean']: 
            file_name = 'c_'+str(c_)+'kernel_'+kernel_sigma+'_num_particles'+str(num_particles)+'_mu_init'+str(mu_init)+'sigma_init'+str(sigma_init)
        else:
            file_name = 'c_'+str(c_)+'kernel_'+str(kernel_sigma)+'_num_particles'+str(num_particles)+'_mu_init'+str(mu_init)+'sigma_init'+str(sigma_init)+'lr='+str(lr)

        with open('./results/APRIL/TUESDAY_04_09__'+file_name+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('DONE!')

'''
import matplotlib.pyplot as plt

for out in results:
    traj,config = out
    kernel_sigma,num_particles,mu_init,sigma_init = config
    
    plt.plot(traj, )


plt.ylabel('some numbers')
plt.show()
'''

'''
##########################
list_lr = [0.1, 0.05, 0.01]
list_num_particles = [10,20,200]
#entr_values

##########################

lr = 0.01
num_particles = 200
num_steps = 800


dim = 2
n_gmm = 1
gmm_std = 1

kernel_sigma = 5

sigma_init = 6
mu_init = 0
sigma_V_init = 1

init_chart, samplers, charts, entr_values = my_experiment(dim=dim, n_gmm=n_gmm, num_particles=num_particles, num_svgd_step=num_steps, kernel_sigma=kernel_sigma, gmm_std=gmm_std, lr=lr, mu_init=mu_init, sigma_init=sigma_init, sigma_V_init=sigma_V_init)
(charts[0][0]|charts[0][10]|charts[0][20]|charts[0][-1]).save('./charts/charts_ld.html')

'''



##############################################################################################################################
'''
sigma_init = 0.2
mu_init = 6
#num_steps = 20

init_chart2, samplers2, charts2, entr_values2 = my_experiment(dim=dim, n_gmm=n_gmm, num_particles=num_particles, num_svgd_step=num_steps, kernel_sigma=kernel_sigma, gmm_std=gmm_std, lr=lr, mu_init=mu_init, sigma_init=sigma_init, sigma_V_init=sigma_V_init)

#alt.vconcat(init_chart | charts_svgd[-1] | charts_ld[-1]| charts_hmc[-1], init_chart2 | charts_svgd2[-1] | charts_ld2[-1]|charts_hmc2[-1])
# init_chart | charts_svgd[-1] | charts_ld[-1]
# charts_svgd[-1]
# init_chart
#init_chart2.save('./charts/init_chart2.html')

(charts2[0][0]|charts2[0][10]|charts2[0][20]|charts2[0][-1]).save('./charts/charts_ld2.html')
'''
##############################################################################################################################
