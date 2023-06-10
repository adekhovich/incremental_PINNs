import argparse
from ipinn import *
from pruning import *

import numpy as np
import os
import random
import torch
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt


from incremental_learning import incremental_learning


def process_data(args, params, tasks, a=0, b=1):
    
    
    X_star_list = []
    u_star_list = [] 
    X_u_train_list = []
    u_train_list = []
    X_f_train_list = [] 
    bc_lb_list, bc_ub_list = [], []
    x_list, t_list = [], []
    
    nt = args.nt
    N_f = args.N_f
    
    beta = params['beta']
    nu = params['nu']
    rho = params['rho']
    
    print(tasks)
    
    u0 = None
    num_tasks = len(tasks)
    for task_id in range(num_tasks):
        
        x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(a, b, nt).reshape(-1, 1)
        X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

        # remove initial and boundaty data from X_star
        t_noinitial = t[1:]
        # remove boundary at x=0
        x_noboundary = x[1:]
        X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
        X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

        # sample collocation points only from the interior (where the PDE is enforced)
        X_f_train = sample_random(X_star_noinitial_noboundary, N_f, seed=args.seed)
        if 'convection' in tasks[task_id] or 'diffusion' in tasks[task_id]:
            u_vals = convection_diffusion(args.u0_str, u0=u0, nu=nu[task_id], beta=beta[task_id], source=args.source, t0=a, T=b, xgrid=args.xgrid, nt=nt)
            G = np.full(X_f_train.shape[0], float(args.source))
        elif 'rd' in tasks[task_id]:
            u_vals = reaction_diffusion_discrete_solution(args.u0_str, u0=u0, nu=nu[task_id], rho=rho[task_id], t0=a, T=b, nx=args.xgrid, nt=nt)
            G = np.full(X_f_train.shape[0], float(args.source))
        elif 'reaction' in tasks[task_id]:
            u_vals = reaction_solution(args.u0_str, u0=u0, rho=rho[task_id], t0=a, T=b, nx=args.xgrid, nt=nt)
            G = np.full(X_f_train.shape[0], float(args.source))
        else:
            print("WARNING: System is not specified.")

        u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
        Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid


        xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
        uu1 = Exact[0:1,:].T # u(x, t) at t=0
        bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
        uu2 = Exact[:,0:1] # u(-end, t)

        xx_T = np.hstack((X[-1:,:].T, T[-1:,:].T))

        # generate the other BC, now at x=2pi
        t = np.linspace(a, b, nt).reshape(-1, 1)
        x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
        bc_ub = np.hstack((x_bc_ub, t))

        u_train = uu1 # just the initial condition
        X_u_train = xx1 # (x,t) for initial condition
        
        #u0=u_star[-args.xgrid:].reshape(-1, )
        
        X_star_list.append(X_star)
        u_star_list.append(u_star)
        X_u_train_list.append(X_u_train)
        u_train_list.append(u_train)
        X_f_train_list.append(X_f_train)
        bc_lb_list.append(bc_lb)
        bc_ub_list.append(bc_ub)
        x_list.append(x)
        t_list.append(t)
    
    
        
    return X_star_list, u_star_list, X_u_train_list, u_train_list, X_f_train_list, bc_lb_list, bc_ub_list, x_list, t_list, G



################
# Arguments
################

def main():
    parser = argparse.ArgumentParser(description='Incremental learning for PINNs')

    parser.add_argument('--system', type=str, default='convection', help='System to study.')
    parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
    parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
    parser.add_argument('--optimizer_name', type=str, default='Adam', help='Optimizer of choice.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--num_epochs', type=int, default=20000, help='Number of training epochs.')

    parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks')
    parser.add_argument('--alpha_fc', type=float, default=0.8, help='Pruning parameter')

    parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

    parser.add_argument('--a', type=float, default=0.0, help='Start of the interval')
    parser.add_argument('--b', type=float, default=1.0, help='End of the interval')
    parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
    parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
    parser.add_argument('--nu', type=float, default=1.0, help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
    parser.add_argument('--rho', type=float, default=1.0, help='reaction coefficient for u*(1-u) term.')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value that scales the du/dx term. 0 if only doing diffusion.')
    parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
    parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")

    parser.add_argument('--layers', type=str, default='50,50,50,50', help='Dimensions/layers of the NN, minus the first layer.')
    parser.add_argument('--net', type=str, default='DNN', help='The net architecture that is to be used.')
    parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
    parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')

    parser.add_argument('--visualize', default=False, help='Visualize the solution.')
    #parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')

    args = parser.parse_args()

    # CUDA support
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    nu = [args.nu]*args.num_tasks
    beta = [args.beta]*args.num_tasks
    rho = [args.rho]*args.num_tasks

    alpha_fc = args.alpha_fc
    seed = args.seed
    lr = args.lr
    num_tasks = args.num_tasks

    if args.system == 'convection':
        tasks = [args.system] *  args.num_tasks
        nu = [0.0]*args.num_tasks
        rho = [0.0]*args.num_tasks

        if num_tasks > 1:
            beta = [1, 10, 20, 30, 40]
 
            
    elif args.system == 'rd': # reaction-diffusion
        beta = [0.0]*args.num_tasks
        if num_tasks > 1:
            tasks = ['reaction', 'diffusion', 'rd']
        else:
            tasks = ['rd']    
    elif args.system == 'dr': # reaction-diffusion
        beta = [0.0]*args.num_tasks
        if num_tasks > 1:
            tasks = ['diffusion', 'reaction', 'rd']
        else:
            tasks = ['rd']

    elif args.system == 'reaction':
        tasks = [args.system] *  args.num_tasks
        beta = [0.0]*args.num_tasks
        nu = [0.0]*args.num_tasks
        
        if num_tasks > 1:
            rho = [1, 2, 3, 4, 5]


    print('nu', nu, 'beta', beta, 'rho', rho)

    params = {
        'beta' : beta,
        'nu' : nu,
        'rho' : rho
    }

    # parse the layers list here
    orig_layers = args.layers
    layers = [int(item) for item in args.layers.split(',')]

    set_seed(args.seed)     

    X_star, u_star, X_u_train, u_train, X_f_train, bc_lb, bc_ub, x, t, G = process_data(args, params, tasks, a=args.a, b=args.b)
    layers.insert(0, X_u_train[0].shape[-1])
    layers.append(1)

    model = iPINN(args=args, tasks=tasks, 
                  X_u_train=X_u_train, u_train=u_train, X_f_train=X_f_train, bc_lb=bc_lb, bc_ub=bc_ub, 
                  layers=layers, G=G, nu=nu, beta=beta, rho=rho,
                  optimizer_name=args.optimizer_name, lr=args.lr, weight_decay=args.weight_decay, net=args.net, num_epochs=args.num_epochs, 
                  L=args.L, activation=args.activation, loss_style=args.loss_style)

    print(model.dnn)
    print("Total parameters: ", model.dnn.total_params())


    model = incremental_learning(args, model, X_star, u_star, X_u_train, u_train, X_f_train, bc_lb, bc_ub, num_tasks, alpha_fc, device)

    error_u_relative_list, error_u_abs_list, error_u_linf_list = validate(model, X_star, u_star)

    print('Mean error u rel: %e' % (np.mean(error_u_relative_list)))
    print('Mean error u abs: %e' % (np.mean(error_u_abs_list)))

    if args.visualize:
        path = f"heatmap_results/{args.system}"
        if not os.path.exists(path):
            os.makedirs(path)

        u_pred = []
        x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(0, 1, args.nt).reshape(-1, 1)

        for task_id in range(model.num_learned):
            x = X_star[task_id]
            model.set_task(task_id)

            u_pred.append(model.predict(x))
            u_pred[task_id] = np.array(u_pred[task_id].reshape(args.nt, args.xgrid))
            u_star[task_id] = np.array(u_star[task_id].reshape(args.nt, args.xgrid))


            u_diff(u_star[task_id], u_pred[task_id], task_id, x, t, nu, beta, rho, seed, layers, args.N_f, args.L, args.source, lr, args.u0_str, tasks[task_id], path=path,
              relative_error=False, save=True)

            exact_u(u_star[task_id], task_id, x, t, nu, beta, rho, layers, args.N_f, args.L, args.source, args.u0_str, tasks[task_id], path=path, save=True)
            
            
    return 0



if __name__ == "__main__":
    main()