import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math
import pdb

import os
path = '/home/local/QCRI/elaabouazza/svgd/MET-P-SVGD/results'
files = os.listdir(path)
files = [f for f in files if f.startswith('THURSDAY_04')]
########################plotting####################
# num_particles = 500
static_kernel = False
'''
files_=[ "results_gmmc_3kernel_0.5_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.5.pickle",\
"results_gmmc_3kernel_0.5_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.3.pickle",\
"results_gmmc_3kernel_0.5_num_particles"+str(num_particles)+"_mu_init0.0sigma_init1.0.pickle",\
"results_gmmc_3kernel_0.5_num_particles"+str(num_particles)+"_mu_init0.0sigma_init2.0.pickle",\
"results_gmmc_3kernel_0.5_num_particles"+str(num_particles)+"_mu_init0.0sigma_init3.0.pickle",\
"results_gmmc_3kernel_1.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.3.pickle",\
"results_gmmc_3kernel_1.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.5.pickle",\
"results_gmmc_3kernel_1.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init1.0.pickle",\
"results_gmmc_3kernel_1.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init2.0.pickle",\
"results_gmmc_3kernel_1.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init3.0.pickle",\
"results_gmmc_3kernel_2.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.3.pickle",\
"results_gmmc_3kernel_2.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.5.pickle",\
"results_gmmc_3kernel_2.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init1.0.pickle",\
"results_gmmc_3kernel_2.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init2.0.pickle",\
"results_gmmc_3kernel_2.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init3.0.pickle",\
"results_gmmc_3kernel_3.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.3.pickle",\
"results_gmmc_3kernel_3.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.5.pickle",\
"results_gmmc_3kernel_3.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init1.0.pickle",\
"results_gmmc_3kernel_3.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init2.0.pickle",\
"results_gmmc_3kernel_3.0_num_particles"+str(num_particles)+"_mu_init0.0sigma_init3.0.pickle"]

files_forth=["results_gmmc_3kernel_forth_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.3.pickle",\
"results_gmmc_3kernel_forth_num_particles"+str(num_particles)+"_mu_init0.0
sigma_init0.5.pickle",\
"results_gmmc_3kernel_forth_num_particles"+str(num_particles)+"_mu_init0.0sigma_init1.0.pickle",\
"results_gmmc_3kernel_forth_num_particles"+str(num_particles)+"_mu_init0.0sigma_init2.0.pickle",\
"results_gmmc_3kernel_forth_num_particles"+str(num_particles)+"_mu_init0.0sigma_init3.0.pickle"]

files_median=["results_gmmc_3kernel_median_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.3.pickle",\
"results_gmmc_3kernel_median_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.5.pickle",\
"results_gmmc_3kernel_median_num_particles"+str(num_particles)+"_mu_init0.0sigma_init1.0.pickle",\
"results_gmmc_3kernel_median_num_particles"+str(num_particles)+"_mu_init0.0sigma_init2.0.pickle",\
"results_gmmc_3kernel_median_num_particles"+str(num_particles)+"_mu_init0.0sigma_init3.0.pickle"]
    
files_mean=["results_gmmc_3kernel_mean_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.3.pickle",\
"results_gmmc_3kernel_mean_num_particles"+str(num_particles)+"_mu_init0.0sigma_init0.5.pickle",\
"results_gmmc_3kernel_mean_num_particles"+str(num_particles)+"_mu_init0.0sigma_init1.0.pickle",\
"results_gmmc_3kernel_mean_num_particles"+str(num_particles)+"_mu_init0.0sigma_init2.0.pickle",\
"results_gmmc_3kernel_mean_num_particles"+str(num_particles)+"_mu_init0.0sigma_init3.0.pickle"]

files = files_ +files_forth+files_median+files_mean
'''
#if static_kernel:
#    files = files_


############################################################################################################
# plot the initial distribution

import altair as alt
alt.data_transformers.enable('default', max_rows=None)

#Helper Functions
def get_density_chart(P, d=4.0, step=0.1):
    
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy).exp().unsqueeze(-1).cpu()

    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),})

    chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('p:Q', scale=alt.Scale(scheme='lightorange', domain=[0, 0.2])),
    tooltip=['x','y','p']).properties(
    width=220,
    height=190)

    return chart

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
            tmp = (- 0.5 * ( torch.matmul(torch.matmul((samples - self.means[i]).unsqueeze(2) , torch.diag(1/self.std[i]).unsqueeze(0).unsqueeze(1)) , (samples - self.means[i]).unsqueeze(-1) )) - 0.5 * np.log(  ((2 * np.pi)**self.dim) * self.std[i].prod() )) + self.mix_probs[i].log()
            logps.append(tmp.squeeze())
            #logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log( 2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
        
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp





########################plotting####################
fig = plt.figure(1)

cnt = 0

data_plt = {'entropy[-3]': np.zeros(len(files)),
    'entropy[-2]': np.zeros(len(files)),
    'entropy[-1]': np.zeros(len(files)),
    'entropy[0]': np.zeros(len(files)),
    'entropy[1]': np.zeros(len(files)),
    'entropy[2]': np.zeros(len(files)),
    'entropy[3]': np.zeros(len(files)),
    'kernel': np.zeros(len(files)),
    'num_particles': np.zeros(len(files)),
    'p0_sigma': np.zeros(len(files))}


c_list = [-3,-2,-1,0,1,2,3]
n_gmm = 5
stds = torch.tensor([[0.16, 1.0], [1.0, 0.16], [0.5, 0.5], [0.5, 0.5], [0.5,0.5]])
#stds = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

mix_probs = (1.0/n_gmm) * torch.ones(n_gmm)


class gauss_distr_(object):
    def __init__(self, mu, sigma):
        self.gauss = torch.distributions.MultivariateNormal(torch.Tensor(mu),covariance_matrix= torch.diag(sigma))
        
    def log_prob(self, x):
        return self.gauss.log_prob(x)

####################### lower bound implementation #######################
def get_entr_lb(c_list, stds, mix_probs):

    def compute_entr_lb(mean_, std_, mix_probs_, n_gmm=5):
        entr_total = 0
        
        for i in range(n_gmm):
            entr_tmp = 0

            for j in range(n_gmm):
                g_dist = gauss_distr_(mean_[j],std_[j] + std_[i])
                entr_tmp += mix_probs_[j] * torch.exp(g_dist.log_prob(mean_[i]))

            entr_total += mix_probs_[i] * torch.log(entr_tmp)

        return - entr_total
    
    list_entr_lb = []

    for i in range(len(c_list)):
        means = torch.tensor([[0.0, 0.0], [3.0, 2.0], [1.0, -0.5], [2.5, 1.5], [c_list[i],c_list[i]]])
        list_entr_lb.append(compute_entr_lb(means, stds, mix_probs))

    #print(list_entr_lb)

    return list_entr_lb

####################### upper bound implementation #######################
def get_entr_ub(c_list, stds, mix_probs):

    def compute_entr_ub(mean_, std_, mix_probs_, n_gmm=5):
        
        entr_total = 0

        for i in range(n_gmm):

            tmp = (2 * np.pi * np.exp(1) ) ** len(mean_[0])

            entr_total += mix_probs_[i] * ( - torch.log(mix_probs_[i]) + 0.5 *  torch.log( tmp * torch.det(torch.diag(std_[i]))))

        return  entr_total


    list_entr_ub = []

    for i in range(len(c_list)):
        means = torch.tensor([[0.0, 0.0], [3.0, 2.0], [1.0, -0.5], [2.5, 1.5], [c_list[i],c_list[i]]])
        list_entr_ub.append(compute_entr_ub(means, stds, mix_probs))

    #print(list_entr_ub)
    return list_entr_ub

####################### gaussian upper bound implementation #######################
def get_gauss_ub(c_list, mix_probs, stds):
    list_gauss_ub = []

    for i in range(len(c_list)):
        #print("__________", c_list[i])
        
        # compute mean
        means = torch.tensor([[0.0, 0.0], [3.0, 2.0], [1.0, -0.5], [2.5, 1.5], [c_list[i],c_list[i]]])
        envelop_g_mean = (mix_probs.view(-1,1) * means).sum(0)

        # compute variance
        envelop_g_cov = 0

        for i in range(n_gmm):
            envelop_g_cov +=  mix_probs[i] * (torch.diag(stds[i]) +  means[i].unsqueeze(-1) * means[i].unsqueeze(0)) 
        
        for k in range(n_gmm):
            for j in range(n_gmm):
                envelop_g_cov -= mix_probs[k] * mix_probs[j] * means[k].unsqueeze(-1) * means[j].unsqueeze(0)

        # initialize a gaussian distr
        gauss = torch.distributions.MultivariateNormal(torch.Tensor(envelop_g_mean),covariance_matrix= envelop_g_cov)
        list_gauss_ub.append(gauss.entropy().item())

        #print("__________mean: ", envelop_g_mean)
        #print("__________var: ", envelop_g_cov)
    return list_gauss_ub


# lower bound
list_entr_lb = get_entr_lb(c_list, stds, mix_probs)

# upper bound
list_entr_ub = get_entr_ub(c_list, stds, mix_probs)

# gauss bound 
list_gauss_ub = get_gauss_ub(c_list, mix_probs, stds)


def plot_curve(traj, kernel_sigma, sigma_init, num_particles):
    try:
        plt.plot(traj, label="$\sigma_k$:"+str(int(kernel_sigma))+'  $\sigma_0$:'+str(sigma_init)+"  M:"+str(num_particles) ,linestyle=linestyle, linewidth=1)
    except:
        plt.plot(traj, label="$\sigma_k$:"+kernel_sigma+'  $\sigma_0$:'+str(sigma_init)+"  M:"+str(num_particles) ,linestyle=linestyle, linewidth=1)
        

c_list = [-3,-2,-1,0,1,2,3]
g_charts=[]

for c_ in c_list:
    gauss = GMMDist(dim=2, n_gmm=5, c_=c_)
    g_charts.append(get_density_chart(gauss, d=4.0, step=0.1)) 


# # (g_charts[0]).save('./figs/init_chart_c0.html')
# (g_charts[1]).save('./figs/init_chart_c1.html')
# (g_charts[2]).save('./figs/init_chart_c2.html')
# (g_charts[3]).save('./figs/init_chart_c3.html')
# (g_charts[4]).save('./figs/init_chart_c4.html')
# (g_charts[5]).save('./figs/init_chart_c5.html')
# (g_charts[6]).save('./figs/init_chart_c6.html')


##############################################################################################################################

# create a dataframe
import pandas as pd
  
# initialize data of lists
og_c_values = [-3, -2, -1, 0, 1, 2, 3]
c_values = og_c_values*len(files)

tmp_data = {'c': og_c_values,
        'Entropy': [0]*len(og_c_values),
        r'$\sigma_{k}$': [0]*len(og_c_values),
        r'$\sigma_{0}$': [0]*len(og_c_values), 
        'num_particles':[0]*len(og_c_values),
        'sigma_init x mu_init':[(0,0)]*len(og_c_values)}
data = {'c': c_values,
        'Entropy': [0]*len(c_values),
        r'$\sigma_{k}$': [0]*len(c_values),
        r'$\sigma_{0}$': [0]*len(c_values), 
        'num_particles':[0]*len(c_values),
        'sigma_init x mu_init':[(0,0)]*len(c_values)}
  
# Create DataFrame
df = pd.DataFrame(data)
  


##############################################################################################################################
index =  0
import seaborn as sns

for file_name in files:
    # open a file, where you stored the pickled data
    try:
        file = open("./results/"+file_name, 'rb')
        #file = open("./results/results_gmmc_3kernel_1.0_num_particles200_mu_init0.0sigma_init2.0.pickle", 'rb')
        #print(file_name)
    except:
        print('!!!!!', file_name)
        #pdb.set_trace()
        continue

    # dump information to that file
    traj,config = pickle.load(file)[0]
    kernel_sigma,num_particles,mu_init,sigma_init = config
    tmp_df = pd.DataFrame(tmp_data)
    try:
        df["Entropy"].iloc[index:index+7]=traj
    except:
        print('******', file_name)
        continue

# __________________________________________________________________________________
    tmp_df['Entropy'] = traj
    tmp_df['num_particles']=[num_particles]*7
    tmp_df[r'$\sigma_{k}$']=[kernel_sigma]*7
    tmp_df[r'$\sigma_{0}$']=[sigma_init]*7
    tmp_df['sigma_init x mu_init']=[(sigma_init, mu_init)]*7
# __________________________________________________________________________________
    df['num_particles'].iloc[index:index+7]=[num_particles]*7
    df[r'$\sigma_{k}$'].iloc[index:index+7]=[kernel_sigma]*7
    df[r'$\sigma_{0}$'].iloc[index:index+7]=[sigma_init]*7
    df['sigma_init x mu_init'].iloc[index:index+7]=[(sigma_init, mu_init)]*7
    index += 7 
# ____________________________________________________________________________________
    g=sns.relplot(data=tmp_df, x='c', y='Entropy', hue=r'$\sigma_{k}$', style=r'$\sigma_{0}$', aspect=1.5, kind="line", palette=["c", "b", "g", "m", "r", "brown", "orange"])


    plt.plot([-3,-2,-1,0,1,2,3],list_entr_lb, label="Lower Bound" ,color='k', linewidth=3)
    plt.plot([-3,-2,-1,0,1,2,3],list_entr_ub, label="Upper Bound" ,color='k', linewidth=3)
    plt.plot([-3,-2,-1,0,1,2,3],list_gauss_ub, label="Gaussian Bound" ,color='y',linewidth=3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    g._legend.remove()
    plt.tight_layout()
    plt.title(f'congifg-- kernel_variance={kernel_sigma}&num_particles={num_particles}&sigma_init={sigma_init}')
    plt.savefig('./results/imgs/THURSDAY_04_04__entr_results_'+str(num_particles)+'___'+str(kernel_sigma)+'___'+f'sigma_init_{sigma_init}x_mu_init_{mu_init}_:'+'.png')
    #df.loc[cnt] = traj[0],traj[1],traj[2],traj[3],traj[4],traj[5],traj[6], kernel_sigma, sigma_init, num_particles

    #print("_______")
    #print("config", config)
    #print(traj)
    # close the file
    '''
    if kernel_sigma==2 or kernel_sigma=="forth":
        linestyle='dashed'
    elif kernel_sigma==3 or kernel_sigma=="median":
        linestyle='dotted'
    else:
        linestyle = '-'
    '''
    traj = np.array(traj)

    if all(traj<np.array(list_gauss_ub)) and all(traj<np.array(list_entr_ub)) and all(traj>np.array(list_entr_lb)):
        print('_______ within bounds _______',file_name)
        #plot_curve(traj, kernel_sigma, sigma_init, num_particles)
    
    file.close()

    cnt += 1


###########################################################################

'''
plt.xlabel("c")
plt.ylabel("entropy")
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.tight_layout()
plt.xticks(np.arange(7), ["-3","-2","-1","0","1","2","3"])

if static_kernel:
    plt.savefig('./figs_new/entr_results_static_kernel_'+str(num_particles)+'.pdf')
else:
    plt.savefig('./figs_new/entr_results_'+str(num_particles)+'.pdf')
'''
####################################################################################################################

#df = (df.melt(id_vars='sigma_0', var_name='Feature', value_name='Value', ignore_index=False))

im = df

g=sns.relplot(data=im, x='c', y='Entropy', hue=r'$\sigma_{k}$', style=r'$\sigma_{0}$', aspect=1.5, kind="line", palette=["c", "b", "g", "m", "r", "brown", "orange"])


plt.plot([-3,-2,-1,0,1,2,3],list_entr_lb, label="Lower Bound" ,color='k', linewidth=3)
plt.plot([-3,-2,-1,0,1,2,3],list_entr_ub, label="Upper Bound" ,color='k', linewidth=3)
plt.plot([-3,-2,-1,0,1,2,3],list_gauss_ub, label="Gaussian Bound" ,color='y',linewidth=3)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

g._legend.remove()
plt.tight_layout()

# if static_kernel:
#     plt.savefig('./results/imgs/NEWentr_results_static_kernel_'+str(num_particles)+'.pdf')
# else:
#     # plt.savefig('./results/imgs/NEWall_results_'+str(num_particles)+'.pdf')
