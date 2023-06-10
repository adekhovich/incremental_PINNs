import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import copy

from torch.autograd import Variable


from choose_optimizer import *

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def init_weights(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, activation, num_inputs=2, num_outputs=1, use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity()
        elif activation == 'tanh':
            self.activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'leaky_relu':
            self.activation = torch.nn.functional.leaky_relu    
        elif activation == 'gelu':
            self.activation = torch.nn.functional.gelu
        elif activation == 'sin':
            self.activation = torch.sin
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            #layer_list.append(('activation_%d' % i, self.activation))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_tasks = 0
        self.task_id = 0
        self.base_masks = self._create_masks(layers, num_inputs)
        
        self.tasks_masks = []
        self.add_mask(task_id=0, num_inputs=num_inputs)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])
        
        self.apply(init_weights)


    def _create_masks(self, layers, num_inputs=2):
        print(layers)
        masks = [torch.ones(layers[1], layers[0]), torch.ones(layers[1])]
        
        for l in range(1, len(layers)-2):
            masks.append(torch.ones(layers[l+1], layers[l]))
            masks.append(torch.ones(layers[l+1]))
        
        masks.append(torch.ones(layers[-1], layers[-2]))
        masks.append(torch.ones(layers[-1]))
        
        return masks
    
    def add_mask(self, task_id, num_inputs=2, num_outputs=1):
        self.num_tasks += 1
        self.tasks_masks.append(copy.deepcopy(self.base_masks))
        

    def total_params(self):
        total_number = 0
        for param_name in list(self.state_dict()):
            param = self.state_dict()[param_name]
            total_number += torch.numel(param[param != 0])

        return total_number


    def total_params_mask(self, task_id):
        total_number_fc = torch.tensor(0, dtype=torch.int32)
        for mask in self.tasks_masks[task_id]:
            total_number_fc += mask.sum().int()

        return total_number_fc.item()
    
    
    def set_masks_union(self):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        for id in range(1, self.num_tasks):
            for l in range(0, len(self.base_masks)):
                self.masks_union[l] = copy.deepcopy( 1*torch.logical_or(self.masks_union[l], self.tasks_masks[id][l]) )

    def set_trainable_masks(self, task_id):
        if task_id > 0:            
            for l in range(len(self.trainable_mask)):
                self.trainable_mask[l] = copy.deepcopy( 1*((self.tasks_masks[task_id][l] + self.masks_union[l]) > 0) ) 
        else:    
            self.trainable_mask = copy.deepcopy(self.tasks_masks[task_id]) 
    
    
    def forward(self, x):
        u = x
        
        for l, layer in enumerate(list(self.layers.children())[0:-1]):
            active_weights = layer.weight*self.tasks_masks[self.task_id][2*l].to(device)
            active_bias = layer.bias*self.tasks_masks[self.task_id][2*l+1].to(device)
            u = F.linear(u, weight=active_weights, bias=active_bias)
            u = self.activation(u)
            
        layer = list(self.layers.children())[-1]
        active_weights = layer.weight*self.tasks_masks[self.task_id][-2].to(device)
        active_bias = layer.bias*self.tasks_masks[self.task_id][-1].to(device)
        
        out = F.linear(u, weight=active_weights, bias=active_bias)
            
        
        return out
    
    def save_masks(self, file_name='net_masks.pt'):
        masks_database = {}
        
        for task_id in range(self.num_tasks):
            masks_database[task_id] = []
            for l in range(len(self.tasks_masks[0])):
                masks_database[task_id].append(self.tasks_masks[task_id][l])

        torch.save(masks_database, file_name)

    def load_masks(self, file_name='net_masks.pt', num_tasks=1):
        masks_database = torch.load(file_name)

        for task_id in range(num_tasks):
            for l in range(len(self.tasks_masks[task_id])):
                self.tasks_masks[task_id][l] = masks_database[task_id][l]
            
            if task_id+1 < num_tasks:
                self._add_mask(task_id+1)
                
        self.set_masks_union()
        


class iPINN():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, args, tasks, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho, optimizer_name, lr,
        weight_decay, net, num_learned=0, num_epochs=1000, L=1, activation='tanh', loss_style='mean'):

        self.args = args
        #self.system = system
        
        self.tasks = tasks
        self.set_data(X_u_train, u_train, X_f_train, bc_lb, bc_ub, G)
        
        self.net = net

        if self.net == 'DNN':
            self.dnn = DNN(layers, activation).to(device)
        else: # "pretrained" can be included in model path
            # the dnn is within the PINNs class
            self.dnn = torch.load(net).dnn

        self.layers = layers
        self.nu = nu
        self.beta = beta
        self.rho = rho

        #self.G = torch.tensor(G, requires_grad=True).float().to(device)
        #self.G = self.G.reshape(-1, 1)

        self.L = L

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name
        
        self.experiment_name = f"{args.system}_{args.activation}_alpha{args.alpha_fc}_wd{args.weight_decay}_{args.num_tasks}tasks_{int(args.a)}-{int(args.b)}_seed{args.seed}"       
        self.num_learned = num_learned        
        
        if optimizer_name == "Adam":
            self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr, (0.9, 0.999), 1e-08, weight_decay, False)
            #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500, verbose=True)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.1)
        else:
            self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)

        self.loss_style = loss_style
        #self.file_name = file_name

        self.iter = 0
        
    def set_data(self, X_u_train, u_train, X_f_train, bc_lb, bc_ub, G=0.0):        
        self.x_u, self.t_u, self.x_f, self.t_f = [], [], [], []
        self.x_bc_lb, self.t_bc_lb, self.x_bc_ub, self.t_bc_ub  = [], [], [], []
        self.u = []
        
        for task_id in range(len(X_u_train)):
            self.x_u.append( torch.tensor(X_u_train[task_id][:, 0:1], requires_grad=True).float().to(device) )
            self.t_u.append( torch.tensor(X_u_train[task_id][:, 1:2], requires_grad=True).float().to(device) )
            self.x_f.append( torch.tensor(X_f_train[task_id][:, 0:1], requires_grad=True).float().to(device) )
            self.t_f.append( torch.tensor(X_f_train[task_id][:, 1:2], requires_grad=True).float().to(device) )
            
            
            self.x_bc_lb.append( torch.tensor(bc_lb[task_id][:, 0:1], requires_grad=True).float().to(device) )
            self.t_bc_lb.append( torch.tensor(bc_lb[task_id][:, 1:2], requires_grad=True).float().to(device) )
            self.x_bc_ub.append( torch.tensor(bc_ub[task_id][:, 0:1], requires_grad=True).float().to(device) )
            self.t_bc_ub.append( torch.tensor(bc_ub[task_id][:, 1:2], requires_grad=True).float().to(device) )
            
            self.u.append(torch.tensor(u_train[task_id], requires_grad=True).float().to(device))
        
        
        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)
        

    def net_u(self, x, t):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]
  
 
        if 'convection' in self.system or 'diffusion' in self.system:
            f = u_t - self.nu[self.dnn.task_id]*u_xx + self.beta[self.dnn.task_id]*u_x #- self.G
        elif 'rd' in self.system:
            f = u_t - self.nu[self.dnn.task_id]*u_xx - self.rho[self.dnn.task_id]*u + self.rho[self.dnn.task_id]*u**2
        elif 'reaction' in self.system:
            f = u_t - self.rho[self.dnn.task_id]*u + self.rho[self.dnn.task_id]*u**2
        return f

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""
               
        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]
            
        
        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x
    
    
    def set_task(self, task_id):
        self.dnn.task_id = task_id
        
        if self.tasks != None:
            self.system = self.tasks[task_id]
        
        return
    
    
    def rewrite_parameters(self, old_params):
        l = 0
        for param, old_param in zip(self.dnn.parameters(), old_params()):
            param.data = param.data*self.dnn.trainable_mask[l].to(device) + old_param.data*(1-self.dnn.trainable_mask[l].to(device))
            l += 1
            
        return   
    
            
    def loss_pinn_one_task(self, task_id):
        
        u_pred = self.net_u(self.x_u[task_id], self.t_u[task_id])
        u_pred_lb = self.net_u(self.x_bc_lb[task_id], self.t_bc_lb[task_id])
        u_pred_ub = self.net_u(self.x_bc_ub[task_id], self.t_bc_ub[task_id])
        if self.nu[task_id] != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb[task_id], self.x_bc_ub[task_id])
            
        
        f_pred = self.net_f(self.x_f[task_id], self.t_f[task_id])
        
        if self.loss_style == 'mean':
            loss_u_t0 = torch.mean((self.u[task_id] - u_pred) ** 2)
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            if self.nu[task_id] != 0:
                loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.mean(f_pred ** 2)
        elif self.loss_style == 'sum':
            loss_u = torch.sum((self.u[task_id] - u_pred) ** 2)
            loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
            if self.nu[task_id] != 0:
                loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.sum(f_pred ** 2)

        loss = loss_u_t0 + loss_b + self.L*loss_f
        
        return loss, loss_u_t0, loss_b, loss_f
    
    
    def loss_pinn(self):
        
        loss_list, loss_u_t0_list, loss_b_list, loss_f_list  = [], [], [], []
        
        old_grads = copy.deepcopy(self.dnn.parameters)
        for grad in old_grads():
            grad.data = torch.zeros_like(grad)
                            
        
        for task_id in range(0, self.num_learned):
            
            self.set_task(task_id)
            
            loss, loss_u_t0, loss_b, loss_f = self.loss_pinn_one_task(task_id)
            
            loss_list.append(loss)
            loss_u_t0_list.append(loss_u_t0)
            loss_b_list.append(loss_b)
            loss_f_list.append(loss_f)
    

        for task_id in range(len(loss_list)):          
            loss_list[task_id].backward(retain_graph=True)          

            l = 0
            for param, old_grad in zip(self.dnn.layers.parameters(), old_grads()):
                param.grad.data = param.grad.data*(self.dnn.tasks_masks[task_id][l]).to(device) + old_grad.data*(1 - self.dnn.tasks_masks[task_id][l]).to(device)   
                old_grad.data = copy.deepcopy(param.grad.data) 
                
                l += 1      
       
                
        loss_tot = torch.tensor(loss_list)
        loss_u_t0_tot = torch.tensor(loss_u_t0_list)
        loss_b_tot = torch.tensor(loss_b_list)
        loss_f_tot = torch.tensor(loss_f_list) 
        
                
        return loss_tot, loss_u_t0_tot, loss_b_tot, loss_f_tot   
                

    def train_step(self, verbose=True):
        
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
                
        loss, loss_u_t0, loss_b, loss_f = self.loss_pinn()    
        
        grad_norm = 0
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_u0: %.5e, loss_b: %.5e, loss_f: %.5e' % (self.iter, grad_norm, loss.sum().item(), loss_u_t0.sum().item(), loss_b.sum().item(), loss_f.sum().item())
                )
            self.iter += 1

        return loss.sum().item()


    def train(self):
        self.dnn.train()
        old_params = copy.deepcopy(self.dnn.parameters)
        min_loss = np.inf
        
        if self.optimizer_name == "Adam":
            self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr, (0.9, 0.999), 1e-08, self.weight_decay, False)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.333, patience=5, verbose=False)
        else:
            self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)
        
        self.iter = 0
        for epoch in range(self.num_epochs):
            self.optimizer.step(self.train_step)
            self.rewrite_parameters(old_params)
                                
            if epoch % 100 == 0:
                loss, loss_u_t0, loss_b, loss_f = self.loss_pinn()
                if min_loss > loss.sum():
                    min_loss = loss.sum().item()
                    torch.save(self.dnn.state_dict(), f"model_{self.experiment_name}.pth")
                   
                self.scheduler.step(loss.sum())
                    
                    
        self.dnn.load_state_dict(torch.load(f"model_{self.experiment_name}.pth"))
    
        

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()

        return u
