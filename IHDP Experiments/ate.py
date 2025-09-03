#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize

from ihdphelpers import truncate_by_g, mse, cross_entropy, truncate_all_by_g
from att import att_estimates


# In[4]:


def _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps):

    h = t * (1./g) - (1.-t) / (1. - g)
    full_lq = (1.-t)*logit(q_t0) + t*logit(q_t1)  # logit predictions from unperturbed model
    logit_perturb = full_lq + eps * h
    return expit(logit_perturb)


# In[5]:


def psi_tmle_bin_outcome(q_t0, q_t1, g, t, y, truncate_level=0.05):
    # solve the perturbation problem

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    eps_hat = minimize(lambda eps: cross_entropy(y, _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps))
                       , 0., method='Nelder-Mead')

    eps_hat = eps_hat.x[0]

    def q1(t_cf):
        return _perturbed_model_bin_outcome(q_t0, q_t1, g, t_cf, eps_hat)

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    return np.mean(ite)


# In[6]:


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05):

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)


    g_loss = mse(g, t)
    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1 # predictions from unperturbed model

    if eps_hat is None:

        eps_hat = np.sum(h*(y-full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

#标准差计算依赖于非参数估计量的渐近展开
    ic = h*(y-q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q-y))
    final_loss = np.mean(np.square(q1(t)-y))


    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss


# In[7]:


def psi_iptw(q_t0, q_t1, g, t, y, truncate_level=0.05):
    ite=(t / g - (1-t) / (1-g))*y
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


# In[8]:


def psi_aiptw(q_t0, q_t1, g, t, y, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    full_q = q_t0 * (1 - t) + q_t1 * t
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    ite = h * (y - full_q) + q_t1 - q_t0

    return np.mean(ite)


# In[9]:


def psi_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


# In[10]:


def psi_weighting(q_t0, q_t1, g, t, y, truncate_level=0.):
    #ite = (q_t1 - q_t0)
    ps = g.copy()
    ps = np.clip(g,a_min = truncate_level,a_max=1-truncate_level)
    ipw = np.mean(t * y / ps - (1-t)*y /(1-ps) )
    dr = ipw - np.mean((t - ps)* q_t1/ps) - np.mean((t - ps) * q_t0 / (1-ps))
    
    return ipw, dr


# In[11]:


def psi_very_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    return y[t == 1].mean() - y[t == 0].mean()


# In[12]:


def ates_from_atts(q_t0, q_t1, g, t, y, truncate_level=0.05):

    prob_t = t.mean()

    att = att_estimates(q_t0, q_t1, g, t, y, prob_t, truncate_level=truncate_level)
    atnott = att_estimates(q_t1, q_t0, 1.-g, 1-t, y, 1.-prob_t, truncate_level=truncate_level)

    att['one_step_tmle'] = att['one_step_tmle'][0]
    atnott['one_step_tmle'] = atnott['one_step_tmle'][0]

    ates = {}
    for k in att.keys():
        ates[k] = att[k]*prob_t + atnott[k]*(1.-prob_t)

    return ates


# In[13]:


def main():
    pass


if __name__ == "__main__":
    main()

