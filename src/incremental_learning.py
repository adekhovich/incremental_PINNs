import torch
import numpy as np

from utils import validate
from pruning import mlp_pruning


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def incremental_learning(args, model, X_star, u_star, X_u_train, u_train, X_f_train, bc_lb, bc_ub, num_tasks, alpha_fc, device):
    interval = np.linspace(args.a, args.b, num_tasks+1)
        
    best_loss = []
    
    for task_id in range(num_tasks): 
        print('-------------------TASK {}------------------------------'.format(task_id+1))
        print(f"Interval: [{interval[0]}, {interval[-1]}]")
                
        model.num_learned += 1
        
        print("TRAINING...")
        model.dnn.set_trainable_masks(task_id)
        model.train()            # train model

                
        if args.num_tasks > 1:  
            print("PRUNING...")
            x_prune = torch.FloatTensor(np.concatenate((X_u_train[task_id], X_f_train[task_id], bc_lb[task_id], bc_ub[task_id]))).to(device)
            model.dnn = mlp_pruning(model.dnn, alpha_fc, x_prune, task_id, device, start_fc_prune=0)    # prune model
            
            print("TRAINING...")
            model.dnn.set_trainable_masks(task_id)
            model.train()            # retrain model

            validate(model, X_star, u_star, save=True)
            model.dnn.set_masks_union()

            model.dnn.save_masks(file_name=f"masks_{model.experiment_name}.pt")
            if task_id + 1 < args.num_tasks:
                model.dnn.add_mask(task_id=task_id+1)
        
    return model
