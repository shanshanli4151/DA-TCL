#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn import metrics, svm, neighbors
import scipy.linalg
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


def kernel(ker, X1, X2=None, gamma=1):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        K = linear_kernel(np.asarray(X1).T, np.asarray(X2).T) if X2 is not None else linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        K = rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma) if X2 is not None else rbf_kernel(np.asarray(X1).T, None, gamma)
    return K
def proxy_a_distance(source_X, target_X):
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int), np.ones(nb_target, dtype=int)))

    clf = svm.LinearSVC(random_state=0)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


# In[ ]:


def estimate_mu(_X1, _Y1, _X2, _Y2):
    adist_m = proxy_a_distance(_X1, _X2)
    C = len(np.unique(_Y1))
    epsilon = 1e-3
    list_adist_c = []
    for i in range(1, C + 1):
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    adist_c = sum(list_adist_c) / C
    mu = adist_c / (adist_c + adist_m)
    mu = min(max(mu, epsilon), 1)
    return mu


# In[ ]:


def BDA_function(Xs, Ys, Xt, Yt, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=10, mode='BDA', estimate_mu_flag=False):
    X = np.hstack((Xs.T, Xt.T))
    X /= np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    C = len(np.unique(Ys))
    H = np.eye(n) - 1 / n * np.ones((n, n))
    Y_tar_pseudo = None
    Xs_new, Xt_new = None, None
    for t in range(T):
        M0 = e @ e.T * C
        N = 0
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in range(1, C + 1):
                if np.isnan(e).any() or np.isinf(e).any():
                    raise ValueError("Vector 'e' contains NaNs or Infs.")
                e = np.zeros((n, 1))
                Ns = len(Ys[np.where(Ys == c)])
                Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                if Ns == 0 or Nt == 0:

                    continue
                if mode == 'WBDA':
                    Ps = Ns / len(Ys)
                    Pt = Nt / len(Y_tar_pseudo)
                    alpha = Pt / Ps
                    mu = 1
                else:
                    alpha = 1

                e[np.where(Ys == c)] = 1 / Ns
                inds = np.where(Y_tar_pseudo == c)[0] + ns
                e[inds] = -alpha / Nt
                e[np.isinf(e)] = 0
                N += np.dot(e, e.T)
                

        if estimate_mu_flag and mode == 'BDA':
            if Xs_new is not None:
                mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
            else:
                mu = 0

        M = (1 - mu) * M0 + mu * N
    
        norm_M = np.linalg.norm(M, 'fro')

        if norm_M < 1e-10:
            print("Warning: Frobenius norm of M is close to zero; skipping normalization.")
        else:
            M /= norm_M  
        if np.isnan(M).any() or np.isinf(M).any():
            raise ValueError("Matrix 'M' contains NaNs or Infs after normalization.")
        #M /= np.linalg.norm(M, 'fro')
        if np.isnan(M).any() or np.isinf(M).any():
            raise ValueError("Matrix 'M' contains NaNs or Infs after construction.")
        K = kernel(kernel_type, X, None, gamma=gamma)
        n_eye = m if kernel_type == 'primal' else n
        if np.isnan(K).any() or np.isinf(K).any():
            raise ValueError("Matrix 'K' contains NaNs or Infs.")
        if np.isnan(M).any() or np.isinf(M).any():
            raise ValueError("Matrix 'M' contains NaNs or Infs.")
        if np.isnan(lamb).any() or np.isinf(lamb):
            raise ValueError("Parameter 'lamb' contains NaNs or Infs.")
        a, b = K @ M @ K.T + lamb * np.eye(n_eye), K @ H @ K.T
        if np.isnan(a).any() or np.isinf(a).any():
            raise ValueError("Matrix 'a' contains NaNs or Infs.")
        if np.isnan(b).any() or np.isinf(b).any():
            raise ValueError("Matrix 'b' contains NaNs or Infs.")
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

        clf = neighbors.KNeighborsRegressor(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        Y_tar_pseudo = clf.predict(Xt_new)

    return Xs_new, Xt_new


# In[ ]:


import os
import glob
import argparse
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ihdp_data import *
from model import *
import json
import numpy as np
from ate import*


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


def run_BDAMerge(data_base_dir='/Users/asus/Desktop/datasets', output_dir='/Users/asus/Desktop/datasets',
                knob_loss=dragonnet_loss_binarycross,
                ratio=1, dragon='', lr2=1e-3, l1_reg=1e-3, batchsize2=64):

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
            
            # Select target indices
            target_col_idx = 3
            target_idx0 = np.where(x[:, target_col_idx] == 0)[0]  # Source domain indices
            target_idx1 = np.where(x[:, target_col_idx] == 1)[0]  # Target domain indices

            # Split source domain and target domain data
            x_s = x[target_idx0]
            y_s = y[target_idx0]
            t_s = t[target_idx0]
            y_cf_s = y_cf[target_idx0]
            mu_0_s = mu_0[target_idx0]
            mu_1_s = mu_1[target_idx0]

            x_t = x[target_idx1]
            y_t = y[target_idx1]
            t_t = t[target_idx1]
            y_cf_t = y_cf[target_idx1]
            mu_0_t = mu_0[target_idx1]
            mu_1_t = mu_1[target_idx1]

            # Perform domain adaptation via BDA
            Xs_new, Xt_new = BDA_function(x_s, y_s, x_t, y_t, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=10, mode='BDA', estimate_mu_flag=True)
            
            # Train on source domain data
            for is_targeted_regularization in [False]:
                print("Is targeted regularization: {}".format(is_targeted_regularization))
                torch.manual_seed(idx)

                if dragon == 'tarnet':
                    print('Creating TarNet model')
                    net = TarNet(x.shape[1]).to(device)

                elif dragon == 'dragonnet':
                    print("Creating DragonNet model")
                    net = DragonNet(x.shape[1]).to(device)

                # Train model using updated x_s (i.e., Xs_new)
                _, _, net, _, _ = train_and_predict_dragons(t_s, y_s, Xs_new, net, seed=idx,
                                                           targeted_regularization=is_targeted_regularization,
                                                           knob_loss=knob_loss, ratio=0, dragon=dragon,
                                                           val_split=val_split, batch_size=batchsize, lr=lr)
                
                # Save base model parameters
                parm = {}
                for name, param in net.named_parameters():
                    param.grad = None
                    parm[name] = param.detach().cpu()  # Ensure parameters are on CPU

                # Transfer learning phase
                if dragon == 'tarnet':
                    print('Creating TarNet_transfer model')
                    net = TarNet_transfer(x.shape[1], parm).to(device)

                elif dragon == 'dragonnet':
                    print("Creating DragonNet_transfer model")
                    net = DragonNet_transfer(x.shape[1], parm).to(device)

                # Perform secondary training on target domain data (using BDA-transformed target domain data)
                test_outputs, train_output, net, train_dict, test_dict = train_and_predict_dragons(
                    t_t, y_t, Xt_new, net, seed=idx, targeted_regularization=is_targeted_regularization,
                    knob_loss=knob_loss, ratio=test_ratio, dragon=dragon,
                    val_split=val_split, batch_size=batchsize2, lr=lr2, l1_reg=l1_reg)

                # Calculate errors
                for data_dict in [train_dict, test_dict]:
                    # Ensure indices are within range
                    max_index = len(mu_1_t) - 1
                    valid_indices = [i for i in data_dict['index'] if 0 <= i <= max_index]
                    
                    if not valid_indices:
                        print(f"Warning: No valid indices in dict, skipping error calculation")
                        continue
                        
                    truth = (mu_1_t[valid_indices] - mu_0_t[valid_indices]).mean()
                    
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
    
    # Save results
    if not os.path.exists(f'./BDAheparams_target{target_col_idx}/'):
        os.makedirs(f'./BDA_merge_params_target{target_col_idx}/')
    
    output_file = f'./BDA_merge_params_target{target_col_idx}/BDAexperiments_transfer_{dragon}_{batchsize2}_{l1_reg}_{lr2}.json'
    
    
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
    print("BDA Error Summary for All Simulations:")
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
        run_BDAMerge(data_base_dir=data_base_dir, output_dir=output_dir, dragon='dragonnet' ,lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)

    if knob == 'tarnet':
        run_BDAMerge(data_base_dir=data_base_dir, output_dir=output_dir, dragon='tarnet',lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)


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
    args, unknown = parser.parse_known_args()
    turn_knob(args.data_base_dir, args.knob, args.output_base_dir,args.transfer_lr, args.l1reg,args.batchsize)


if __name__ == '__main__':
    main()

