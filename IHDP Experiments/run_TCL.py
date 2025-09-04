#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf#将损失放进np.inf
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


print("Forcing CPU-only mode")
def run_ihdp(data_base_dir='/Users/asus/Desktop/datasets', output_dir='/Users/asus/Desktop/result2',
             knob_loss=dragonnet_loss_binarycross,
             ratio=1., dragon='', lr2=1e-3, l1_reg=1e-3, batchsize2=16):

    print("the dragon is {}".format(dragon))
    warnings.filterwarnings("ignore")
    
    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    final_output = []
    
    # Initialize lists to collect all test errors
    all_err_test = []
    all_dr_err_test = []
    all_ipw_error_test = []

    for idx, simulation_file in enumerate(simulation_files):
        try:
            print(f"\nProcessing file {idx+1}/{len(simulation_files)}: {os.path.basename(simulation_file)}")
            
            # Load data
            x = load_and_format_covariates_ihdp(simulation_file)
            t, y, y_cf, mu_0, mu_1 = load_all_other_crap(simulation_file)
            
            
            if len(x) == 0 or len(t) == 0 or len(y) == 0:
                print(f"Skipping empty dataset: {simulation_file}")
                continue
                
            ## Hyperparameter settings
            batchsize = 16
            lr = 1e-3
            test_ratio = 0.5
            val_split = 0.3
            batchsize2 = batchsize2
            lr2 = lr2
            l1_reg = l1_reg
            
            ############### Select target indices ################
            target_col_idx = 3
            print(f"Selecting target index: {target_col_idx}")
            
            # Ensure indices are valid
            if target_col_idx >= x.shape[1]:
                print(f"Invalid target_col_idx {target_col_idx} for x with {x.shape[1]} columns")
                continue
                
            target_idx0 = np.where(x[:, target_col_idx] == 0)[0]
            target_idx1 = np.where(x[:, target_col_idx] == 1)[0]
            
            if len(target_idx0) == 0 or len(target_idx1) == 0:
                print(f"Skipping dataset with empty group: {simulation_file}")
                continue
                
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

            for is_targeted_regularization in [False]:
                ## Step 1: Source domain training ####
                print("Is targeted regularization: {}".format(is_targeted_regularization))
                
                # Check source domain data
                if len(x_s) == 0:
                    print("Source domain is empty, skipping")
                    continue
                
                # Create model
                if dragon == 'tarnet':
                    print('Creating TarNet model')
                    net = TarNet(x_s.shape[1]).to(device)
                elif dragon == 'dragonnet':
                    print("Creating DragonNet model")
                    net = DragonNet(x_s.shape[1]).to(device)
                else:
                    print(f"Unknown model type: {dragon}")
                    continue
                    
                # Train on source domain
                print("Training on source domain...")
                try:
                    _, _, net, _, _ = train_and_predict_dragons(
                        t_s, y_s, x_s, net, seed=idx,
                        targeted_regularization=is_targeted_regularization,
                        knob_loss=knob_loss, ratio=0, dragon=dragon,
                        val_split=val_split, batch_size=min(batchsize, len(x_s)), lr=lr)
                except Exception as e:
                    print(f"Error during source domain training: {str(e)}")
                    continue
                
             
                parm = {}
                for name, param in net.named_parameters():
                    if param is not None:
                        param.grad = None
                        # Ensure parameters are CPU tensors
                        parm[name] = param.detach().cpu()
                
                # Initialize transfer learning model
                if dragon == 'tarnet':
                    print('Creating TarNet_transfer model')
                    net = TarNet_transfer(x_t.shape[1], parm).to(device)
                elif dragon == 'dragonnet':
                    print("Creating DragonNet_transfer model")
                    net = DragonNet_transfer(x_t.shape[1], parm).to(device)
                
                ## Step 2: Target domain training and testing ###
                if len(x_t) == 0:
                    print("Target domain is empty, skipping")
                    continue
                
                print("Training on target domain...")
                try:
                    test_outputs, train_output, net, train_dict, test_dict = train_and_predict_dragons(
                        t_t, y_t, x_t, net, seed=idx,
                        targeted_regularization=is_targeted_regularization,
                        knob_loss=knob_loss, ratio=test_ratio, dragon=dragon,
                        val_split=val_split, batch_size=min(batchsize2, len(x_t)), lr=lr2, l1_reg=l1_reg)
                except Exception as e:
                    print(f"Error during target domain training: {str(e)}")
                    continue
                
               
                if not isinstance(train_dict, dict) or not isinstance(test_dict, dict):
                    print("Invalid train_dict or test_dict, skipping")
                    continue
                
              
                train_dict = convert_to_serializable(train_dict)
                test_dict = convert_to_serializable(test_dict)
                    
                # Calculate and record training and test set errors
                for dict_type, data_dict in zip(['train', 'test'], [train_dict, test_dict]):
                    if 'index' not in data_dict or len(data_dict['index']) == 0:
                        print(f"Invalid index in {dict_type} dict")
                        continue
                        
                    max_index = len(mu_1_t) - 1
                    valid_indices = [i for i in data_dict['index'] if 0 <= i <= max_index]
                    
                    if not valid_indices:
                        print(f"No valid indices in {dict_type} dict")
                        continue
                        
                    truth = (mu_1_t[valid_indices] - mu_0_t[valid_indices]).mean()
                    
                    if 'psi_n' not in data_dict or 'dr_n' not in data_dict or 'ipw_n' not in data_dict:
                        print(f"Missing prediction values in {dict_type} dict")
                        continue
                        
                    data_dict['err'] = abs(truth - data_dict['psi_n']).mean()
                    data_dict['dr_err'] = abs(truth - data_dict['dr_n']).mean()
                    data_dict['ipw_error'] = abs(truth - data_dict['ipw_n']).mean()
                    
                    if dict_type == 'test':
                        all_err_test.append(data_dict['err'])
                        all_dr_err_test.append(data_dict['dr_err'])
                        all_ipw_error_test.append(data_dict['ipw_error'])
                
               
                train_dict_formatted = {}
                for k, v in train_dict.items():
                    if 'index' in k:
                        train_dict_formatted[f'{k}_train'] = v
                    else:
                        train_dict_formatted[f'{k}_train'] = v
                
                test_dict_formatted = {}
                for k, v in test_dict.items():
                    if 'index' in k:
                        test_dict_formatted[f'{k}_test'] = v
                    else:
                        test_dict_formatted[f'{k}_test'] = v
                
                combined_dict = {**train_dict_formatted, **test_dict_formatted}
                combined_dict['sim_idx'] = idx
                
              
                combined_dict = convert_to_serializable(combined_dict)
                final_output.append(combined_dict)
                
               
                print(f"Simulation {idx} results:")
                print(f"  Test err: {combined_dict.get('err_test', 'N/A'):.4f}")
                print(f"  Test dr_err: {combined_dict.get('dr_err_test', 'N/A'):.4f}")
                print(f"  Test ipw_error: {combined_dict.get('ipw_error_test', 'N/A'):.4f}")
                
                
                del net, x_s, y_s, t_s, x_t, y_t, t_t
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except Exception as e:
            print(f"Error processing {simulation_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate error statistics for 30 datasets
    if all_err_test:
        err_mean = float(np.mean(all_err_test))
        err_var = float(np.var(all_err_test))
        dr_err_mean = float(np.mean(all_dr_err_test))
        dr_err_var = float(np.var(all_dr_err_test))
        ipw_err_mean = float(np.mean(all_ipw_error_test))
        ipw_err_var = float(np.var(all_ipw_error_test))
    else:
        err_mean = err_var = dr_err_mean = dr_err_var = ipw_err_mean = ipw_err_var = -1
        print("WARNING: No valid test results were collected")
    
    # Add summary statistics to output
    summary = {
        'err_mean': err_mean,
        'err_variance': err_var,
        'dr_err_mean': dr_err_mean,
        'dr_err_variance': dr_err_var,
        'ipw_err_mean': ipw_err_mean,
        'ipw_err_variance': ipw_err_var,
        'successful_runs': len(all_err_test)
    }
    
    # Ensure summary dictionary is serializable
    summary = convert_to_serializable(summary)
    final_output.append({'summary': summary})
    
    # Save results
    if not os.path.exists(f'./TCL-params_target{target_col_idx}/'):
        os.makedirs(f'./TCL-params_target{target_col_idx}/')
    
    output_file = f'./TCL-params_target{target_col_idx}/experiments_transfer_{dragon}_{batchsize2}_{l1_reg}_{lr2}.json'
    
    
    with open(output_file, 'w') as fp:
        # Finally ensure the entire output is serializable
        final_output = convert_to_serializable(final_output)
        json.dump(final_output, fp, indent=2, cls=NumpyEncoder)
    
    # Print summary results
    print("\n" + "="*30)
    print("Error Summary:")
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
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='dragonnet' ,lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)

    if knob == 'tarnet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='tarnet',lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD", default="/Users/asus/Desktop/datasets")
    parser.add_argument('--knob', type=str, default='tarnet',
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

