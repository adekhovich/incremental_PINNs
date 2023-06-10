import numpy as np
import random
import torch

from systems_pbc import *
import pickle

def set_seed(seed):
    print("SEED: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True    
        
    return    

def sample_random(X_all, N, seed):
    """Given an array of (x,t) points, sample N points from this."""
    #set_seed(seed) # this can be fixed for all N_f

    idx = np.random.choice(X_all.shape[0], N, replace=False)
    X_sampled = X_all[idx, :]

    return X_sampled

def set_activation(activation):
    if activation == 'identity':
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return  nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        print("WARNING: unknown activation function!")
        return -1


def error(u_pred, u_star):
    error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
    error_u_abs = np.mean(np.abs(u_star - u_pred))
    error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)
    
    return error_u_relative, error_u_abs, error_u_linf


def validate(model, X_star, u_star, save=False):
    error_u_relative_list, error_u_abs_list, error_u_linf_list = [], [], []
    
    file_name = model.experiment_name
        
    for task_id in range(model.num_learned):
        x = X_star[task_id]
        model.set_task(task_id)
            
        u_pred = model.predict(x)
        
        error_u_relative, error_u_abs, error_u_linf = error(u_pred, u_star[task_id])
        error_u_relative_list.append(error_u_relative)
        error_u_abs_list.append(error_u_abs)
        error_u_linf_list.append(error_u_linf)
        
    print('Error u rel: ', error_u_relative_list)
    print('Error u abs: ' , error_u_abs_list)
    print('Error u linf: ', error_u_linf_list)
    
    if save: 
        if model.num_learned > 1:
            with open(f'{file_name}.pickle', 'rb') as handle:
                error_dict = pickle.load(handle)
        else:
            error_dict = {'err_rel' : [], 'err_abs' : [], 'err_linf' : []}
            
        error_dict['err_rel'].append(np.array(error_u_relative_list))
        error_dict['err_abs'].append(np.array(error_u_abs_list))
        error_dict['err_linf'].append(np.array(error_u_linf_list))

        with open(f'{file_name}.pickle', 'wb') as handle:
            pickle.dump(error_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    return error_u_relative_list, error_u_abs_list, error_u_linf_list

        
