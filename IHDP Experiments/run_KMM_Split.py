#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import argparse
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import *
from ihdp_data import *
import json
import numpy as np
from ate import*


# In[ ]:


import pandas as pd
import sklearn.metrics
from cvxopt import matrix, solvers


# In[ ]:


def kernel(ker, X1, X2, gamma):
    """
    Kernel function to compute kernel matrix based on kernel type.
    :param ker: 'linear' | 'rbf'
    :param X1: First dataset (Xs or Xt)
    :param X2: Second dataset (Xs or Xt)
    :param gamma: Kernel bandwidth (only used for 'rbf')
    :return: Computed kernel matrix
    """
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


# In[ ]:


class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps is None:
            self.eps = self.B / np.sqrt(ns)
        
        # Compute kernel matrix
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)
        
        # Set up and solve the quadratic programming problem
        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta


# In[ ]:


def apply_kmm(Xs, Ys, Xt, Yt, kernel_type='rbf', gamma=1.0, B=1.0):
    """
    Apply KMM to source and target domain data to compute new source data.
    :param Xs: Source data (ns * dim)
    :param Ys: Source labels (ns * 1)
    :param Xt: Target data (nt * dim)
    :param Yt: Target labels (nt * 1)
    :param kernel_type: 'linear' | 'rbf', default is 'rbf'
    :param gamma: Bandwidth parameter for 'rbf' kernel, default is 1.0
    :param B: Bound for beta, default is 1.0
    :return: New source data Xs_new after applying KMM
    """
    # Initialize KMM model
    kmm = KMM(kernel_type=kernel_type, gamma=gamma, B=B)
    
    # Fit KMM model to compute the coefficients
    beta = kmm.fit(Xs, Xt)
    
    # Compute the new source data Xs_new by scaling the original Xs with beta
    Xs_new = beta * Xs
    
    return Xs_new


# In[ ]:


def get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps, truncate_level=0.01):
    """
    getting the back door adjustment & TMLE estimation
    """

    psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
    ipw_n, dr_n = psi_weighting(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                              y_dragon,
                                                                                              truncate_level=truncate_level)
    return psi_n, psi_tmle, initial_loss, final_loss, g_loss,ipw_n, dr_n


# In[ ]:


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# In[ ]:


def _split_output(yt_hat, t, y, y_scaler, x, index):
    """     yt_hat: Generated prediction
            t: Binary treatment assignments
            y: Treatment outcomes
            y_scaler: Scaled treatment outcomes
            x: Covariates
    """
    yt_hat = yt_hat.detach().cpu().numpy()
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


# In[ ]:


def train(train_loader, net, optimizer, criterion,valid_loader= None,l1_reg = None):

    avg_loss = 0

    for i, data in enumerate(train_loader):
  
        inputs, labels = data


        optimizer.zero_grad()


        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if l1_reg is not None:
            l1_penalty = l1_reg * sum([p.abs().sum() for p in net.parameters()])
            loss+= l1_penalty
        loss.backward()
        optimizer.step()

        avg_loss += loss

    valid_loss = None
    if valid_loader is not None:
        valid_loss = 0.0
        net.eval()     
        for data, labels in valid_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            target = net(data)
            loss = criterion(target,labels)
            if l1_reg is not None:
                loss+= l1_reg * sum([p.abs().sum() for p in net.parameters()]) 
            valid_loss += loss
        valid_loss = valid_loss/len(valid_loader)
    return avg_loss / len(train_loader), valid_loss


# In[ ]:


def train_and_predict_dragons(t, y_unscaled, x, net,seed = 0, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', val_split=0.2, batch_size=64,lr =1e-3,l1_reg = None):
    """
    Method for training dragonnet and tarnet and predicting new results
    """    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    verbose = 0
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_unscaled)
    train_outputs = []
    test_outputs = []
    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss


    i = seed
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

    if ratio == 0:
        train_index = np.arange(x.shape[0])
        test_index = train_index
    else:
        train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=ratio, random_state=seed)
        print(f'test_index {test_index}')
   
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)

    yt_test = np.concatenate([y_test, t_test], 1)

    # Create data loader to pass onto training method
    tensors_train = torch.from_numpy(x_train).float().to(device), torch.from_numpy(yt_train).float().to(device)
    train_size = int((val_split) * len(TensorDataset(*tensors_train)))
    val_size = int(len(TensorDataset(*tensors_train))-train_size)
    train_set, valid_set = random_split(TensorDataset(*tensors_train),[train_size,val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=500)

    import time;
    start_time = time.time()

    epochs1 = 100
    epochs2 = 300

    optimizer_Adam = optim.Adam([{'params': net.representation_block.parameters()},
                                 {'params': net.t_predictions.parameters()},
                                 {'params': net.t0_head.parameters(), 'weight_decay': 0.01},
                                 {'params': net.t1_head.parameters(), 'weight_decay': 0.01}], lr=lr)
    optimizer_SGD = optim.SGD([{'params': net.representation_block.parameters()},
                               {'params': net.t_predictions.parameters()},
                               {'params': net.t0_head.parameters(), 'weight_decay': 0.01},
                               {'params': net.t1_head.parameters(), 'weight_decay': 0.01}], lr=lr*0.01, momentum=0.9)
    scheduler_Adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_Adam, mode='min', factor=0.5, patience=5,
                                                          threshold=1e-8, cooldown=0, min_lr=0)
    scheduler_SGD = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_SGD, mode='min', factor=0.5, patience=5,
                                                         threshold=0, cooldown=0, min_lr=0)

    train_loss = 0

    early_stopper = EarlyStopper(patience=2, min_delta=0.)

    # Adam training run
    for epoch in range(epochs1):

        # Train on data
        train_loss,val_loss = train(train_loader, net, optimizer_Adam, loss,valid_loader = valid_loader,l1_reg = l1_reg)
        
        if early_stopper.early_stop(val_loss):             
            break

        scheduler_Adam.step(val_loss)

    print(f"Adam loss: train -- {train_loss}, validation -- {val_loss}, epoch {epoch}")

    # SGD training run
    
    early_stopper = EarlyStopper(patience=40, min_delta=0.)

    for epoch in range(epochs2):
        # Train on data
        train_loss,val_loss = train(train_loader, net, optimizer_SGD, loss,valid_loader = valid_loader,l1_reg = l1_reg)

        if early_stopper.early_stop(val_loss):             
            break
        scheduler_SGD.step(val_loss)
        

    print(f"SGD loss: train --  {train_loss}, validation -- {val_loss},  epoch {epoch}")

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    yt_hat_test = net(torch.from_numpy(x_test).float().to(device))
    yt_hat_train = net(torch.from_numpy(x_train).float().to(device))

    test_outputs += [_split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]
    train_outputs += [_split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
   
    train_all_dicts = _split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)
    test_all_dicts = _split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)
   
    psi_n, psi_tmle, initial_loss, final_loss, g_loss,ipw_n, dr_n = get_estimate(train_all_dicts['q_t0'].reshape(-1, 1), train_all_dicts['q_t1'].reshape(-1, 1), train_all_dicts['g'].reshape(-1, 1), train_all_dicts['t'].reshape(-1, 1), train_all_dicts['y'].reshape(-1, 1), train_all_dicts['index'].reshape(-1, 1), train_all_dicts['eps'].reshape(-1, 1),truncate_level=0.01)

    train_dict = {'psi_n':psi_n, 'classification_mse': g_loss,'ipw_n':ipw_n, 'dr_n':dr_n,'regression_loss':regression_loss(torch.tensor(yt_train).to(device),yt_hat_train).cpu().detach(),'BCE':binary_classification_loss(torch.tensor(yt_train).float().to(device),yt_hat_train).cpu().detach().numpy(),'regression_mse':initial_loss,'index':train_all_dicts['index']}
    
    psi_n, psi_tmle, initial_loss, final_loss, g_loss,ipw_n, dr_n = get_estimate(test_all_dicts['q_t0'].reshape(-1, 1), test_all_dicts['q_t1'].reshape(-1, 1), test_all_dicts['g'].reshape(-1, 1), test_all_dicts['t'].reshape(-1, 1), test_all_dicts['y'].reshape(-1, 1), test_all_dicts['index'].reshape(-1, 1), test_all_dicts['eps'].reshape(-1, 1),truncate_level=0.01)

    
    test_dict = {'psi_n':psi_n, 'classification_mse': g_loss,'ipw_n':ipw_n, 'dr_n':dr_n,'regression_loss':regression_loss(torch.tensor(yt_test).to(device),yt_hat_test).cpu().detach(),'BCE':binary_classification_loss(torch.tensor(yt_test).float().to(device),yt_hat_test).cpu().detach().numpy(),'regression_mses':initial_loss,'index':test_all_dicts['index']}

    return test_outputs, train_outputs, net,train_dict,test_dict


# In[ ]:


def run_KMMsplit(data_base_dir='/Users/asus/Desktop/datasets', output_dir='/Users/asus/Desktop/datasets',
                 knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', lr2=1e-3, l1_reg=1e-3, batchsize2=64):

    print("the dragon is {}".format(dragon))
    device = torch.device("cpu")
    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    
    # Initialize lists to collect all test errors
    all_err_test = []
    all_dr_err_test = []
    all_ipw_error_test = []
    
    final_output = []
    for idx, simulation_file in enumerate(simulation_files):
        try:
            print(f"\nProcessing file {idx+1}/{len(simulation_files)}: {os.path.basename(simulation_file)}")
            
            # Load features and other variables
            x = load_and_format_covariates_ihdp(simulation_file)
            t, y, y_cf, mu_0, mu_1 = load_all_other_crap(simulation_file)
            
            # Hyperparameter settings
            batchsize = 64
            lr = 1e-3
            test_ratio = 0.5
            val_split = 0.3
            batchsize2 = batchsize2
            lr2 = lr2
            l1_reg = l1_reg
            
            # Classify based on binary column t
            # Ensure t is a 1D array and only used for row indexing
            x_t0, y_t0 = x[t.ravel() == 0], y[t.ravel() == 0]
            x_t1, y_t1 = x[t.ravel() == 1], y[t.ravel() == 1]

            # Select target indices
            target_col_idx = 3
            target_idx0 = np.where(x[:, target_col_idx] == 0)[0]  # Source domain indices
            target_idx1 = np.where(x[:, target_col_idx] == 1)[0]  # Target domain indices

            # Get target indices for x_t0 and x_t1
            target_idx0_t0 = np.where(x_t0[:, target_col_idx] == 0)[0]  # Source domain indices in x_t0
            target_idx1_t0 = np.where(x_t0[:, target_col_idx] == 1)[0]  # Target domain indices in x_t0
            target_idx0_t1 = np.where(x_t1[:, target_col_idx] == 0)[0]  # Source domain indices in x_t1
            target_idx1_t1 = np.where(x_t1[:, target_col_idx] == 1)[0]  # Target domain indices in x_t1

            # Split source and target domain data by indices
            x_t0s, y_t0s = x_t0[target_idx0_t0], y_t0[target_idx0_t0]
            x_t0t, y_t0t = x_t0[target_idx1_t0], y_t0[target_idx1_t0]
            x_t1s, y_t1s = x_t1[target_idx0_t1], y_t1[target_idx0_t1]
            x_t1t, y_t1t = x_t1[target_idx1_t1], y_t1[target_idx1_t1]

            # Perform domain adaptation using KMM to get x_t0s_new and x_t1s_new respectively
            x_t0s_new = apply_kmm(x_t0s, y_t0s, x_t0t, y_t0t, kernel_type='rbf', gamma=1.0, B=1.0)
            x_t1s_new = apply_kmm(x_t1s, y_t1s, x_t1t, y_t1t, kernel_type='rbf', gamma=1.0, B=1.0)

            # Merge new source domain data
            Xs_new = np.vstack((x_t0s_new, x_t1s_new))

            # Train model using new source domain data
            for is_targeted_regularization in [False]:
                print("Is targeted regularization: {}".format(is_targeted_regularization))
                torch.manual_seed(idx)

                if dragon == 'tarnet':
                    print('Creating TarNet model')
                    net = TarNet(x.shape[1]).to(device)

                elif dragon == 'dragonnet':
                    print("Creating DragonNet model")
                    net = DragonNet(x.shape[1]).to(device)

                # Train model using updated Xs_new
                _, _, net, _, _ = train_and_predict_dragons(t[target_idx0], y[target_idx0], Xs_new, net, seed=idx,
                                                            targeted_regularization=is_targeted_regularization,
                                                            knob_loss=knob_loss, ratio=0, dragon=dragon,
                                                            val_split=val_split, batch_size=batchsize, lr=lr)

                # Save base model parameters
                parm = {}
                for name, param in net.named_parameters():
                    param.grad = None
                    parm[name] = param.detach().cpu()  # Ensure parameters are on CPU

                # Transfer learning phase, using saved base model parameters
                if dragon == 'tarnet':
                    print('Creating TarNet_transfer model')
                    net = TarNet_transfer(x.shape[1], parm).to(device)

                elif dragon == 'dragonnet':
                    print("Creating DragonNet_transfer model")
                    net = DragonNet_transfer(x.shape[1], parm).to(device)

                # Perform secondary training on target domain data
                test_outputs, train_output, net, train_dict, test_dict = train_and_predict_dragons(
                    t[target_idx1], y[target_idx1], x[target_idx1], net, seed=idx,
                    targeted_regularization=is_targeted_regularization, knob_loss=knob_loss, ratio=test_ratio,
                    dragon=dragon, val_split=val_split, batch_size=batchsize2, lr=lr2, l1_reg=l1_reg)

                # Calculate errors
                for data_dict in [train_dict, test_dict]:
                    # Ensure indices are within range
                    max_index = len(mu_1) - 1
                    valid_indices = [i for i in data_dict['index'] if 0 <= i <= max_index]
                    
                    if not valid_indices:
                        print(f"Warning: No valid indices in dict, skipping error calculation")
                        continue
                        
                    truth = (mu_1[valid_indices] - mu_0[valid_indices]).mean()
                    
                    # Ensure prediction values exist
                    if 'psi_n' not in data_dict or 'dr_n' not in data_dict or 'ipw_n' not in data_dict:
                        print(f"Warning: Missing prediction values in dict, skipping error calculation")
                        continue
                        
                    data_dict['err'] = abs(truth - data_dict['psi_n']).mean()
                    data_dict['dr_err'] = abs(truth - data_dict['dr_n']).mean()
                    data_dict['ipw_error'] = abs(truth - data_dict['ipw_n']).mean()
                    
                    # If it's test set, collect errors for final statistics
                    if data_dict is test_dict:
                        all_err_test.append(data_dict['err'])
                        all_dr_err_test.append(data_dict['dr_err'])
                        all_ipw_error_test.append(data_dict['ipw_error'])
                
                # Convert indices to lists
                test_dict['index'] = test_dict['index'].tolist()
                train_dict['index'] = train_dict['index'].tolist()
                
               
                train_dict_formatted = {f'{k}_train': v.item() if 'index' not in k else v for k, v in train_dict.items()}
                test_dict_formatted = {f'{k}_test': v.item() if 'index' not in k else v for k, v in test_dict.items()}
                
                combined_dict = {**train_dict_formatted, **test_dict_formatted}
                combined_dict['sim_idx'] = idx
                final_output.append(combined_dict)
                
                
                print(f"Simulation {idx} results:")
                print(f"  Test err: {combined_dict.get('err_test', 'N/A'):.4f}")
                print(f"  Test dr_err: {combined_dict.get('dr_err_test', 'N/A'):.4f}")
                print(f"  Test ipw_error: {combined_dict.get('ipw_error_test', 'N/A'):.4f}")

        except Exception as e:
            print(f"Error processing {simulation_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate error statistics across all datasets
    if all_err_test:
        err_mean = np.mean(all_err_test)
        err_var = np.var(all_err_test)
        dr_err_mean = np.mean(all_dr_err_test)
        dr_err_var = np.var(all_dr_err_test)
        ipw_err_mean = np.mean(all_ipw_error_test)
        ipw_err_var = np.var(all_ipw_error_test)
    else:
        # If no successful tests exist, set default values
        err_mean = err_var = dr_err_mean = dr_err_var = ipw_err_mean = ipw_err_var = -1
        print("WARNING: No valid test results were collected")
    
    # Add summary statistics to output
    summary = {
        'err_mean': float(err_mean),
        'err_variance': float(err_var),
        'dr_err_mean': float(dr_err_mean),
        'dr_err_variance': float(dr_err_var),
        'ipw_err_mean': float(ipw_err_mean),
        'ipw_err_variance': float(ipw_err_var),
        'successful_runs': len(all_err_test)
    }
    final_output.append({'summary': summary})
    
   
    if not os.path.exists(f'./KMM-split-params_target{target_col_idx}/'):
        os.makedirs(f'./KMM-split-params_target{target_col_idx}/')
    
    output_file = f'./KMM-split-params_target{target_col_idx}/experiments_transfer_{dragon}_{batchsize2}_{l1_reg}_{lr2}.json'
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(output_file, 'w') as fp:
        json.dump(final_output, fp, indent=2, cls=NumpyEncoder)
    
    print("\n" + "="*30)
    print("Error Summary for All Simulations:")
    print(f"Successful runs: {len(all_err_test)}/{len(simulation_files)}")
    print(f"ATE Error: Mean = {err_mean:.4f}, Variance = {err_var:.4f}")
    print(f"DR Error: Mean = {dr_err_mean:.4f}, Variance = {dr_err_var:.4f}")
    print(f"IPW Error: Mean = {ipw_err_mean:.4f}, Variance = {ipw_err_var:.4f}")
    print(f"Results saved to {output_file}")
    print("="*30 + "\n")

    return final_output


# In[ ]:


def turn_knob(data_base_dir='/Users/asus/Desktop/datasets/', knob='dragonnet',
              output_base_dir='',lr  = 1e-3, l1reg = 1e-4,batchsize = 64):
    output_dir = os.path.join(output_base_dir, knob)

    if knob == 'dragonnet':
        run_KMMsplit(data_base_dir=data_base_dir, output_dir=output_dir, dragon='dragonnet' ,lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)

    if knob == 'tarnet':
        run_KKMMsplit(data_base_dir=data_base_dir, output_dir=output_dir, dragon='tarnet',lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD", default="/Users/asus/Desktop/datasets")
    parser.add_argument('--knob', type=str, default='dragonnet',
                        help="dragonnet or tarnet")

    parser.add_argument('--output_base_dir', type=str, help="directory to save the output",default="/Users/asus/Desktop/datasets")

    parser.add_argument('--transfer_lr',type = float,default=0.001)

    parser.add_argument('--l1reg',type = float,default=0.01)

    parser.add_argument('--batchsize',type = int,default=64)
    #args = parser.parse_args(args=[])
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    turn_knob(args.data_base_dir, args.knob, args.output_base_dir,args.transfer_lr, args.l1reg,args.batchsize)


if __name__ == '__main__':
    main()

