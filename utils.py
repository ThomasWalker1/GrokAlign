## https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split
import torchvision
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regularization'))
import numpy as np
from orthograd import OrthoGrad
import math

def unique_random_combinations(num_features, num_samples):
    seen = set()
    domain = [0, 1]

    if num_samples> 2**num_features:
        print(f"Number of samples > Possible combinations, setting num_samples to {2**num_features}")
        num_samples = 2**num_features
    while len(seen) < num_samples:
        combination = tuple(random.choice(domain) for _ in range(num_features))
        if combination not in seen:
            seen.add(combination)
            yield combination

class SparseParityDataset(Dataset):
    def __init__(self, num_features, num_noise_features, num_samples=None):
        self.num_features = num_features
        self.num_noise_features = num_noise_features
        self.num_samples = num_samples
    
        self.data = torch.tensor(list(unique_random_combinations(num_features + self.num_noise_features, self.num_samples)))
        self.targets = (self.data[:,:num_features].sum(dim=1)%2).float()
        self.targets = self.targets.long()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True, non_linearity=nn.ReLU):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activations = []
        self.activations_from_abs_input = None
        self.embedding = None
        
        self.layers = nn.ModuleList()
        self.non_linearity = non_linearity()
        self.uses_bias = bias
        self.alpha = 1
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(hidden_sizes)+1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias))
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.layers):
            x = self.non_linearity(layer(x)) if i<len(self.layers) -1 else layer(x)
        return x*self.alpha

def softmax_cross_entropy(logits, labels, reduction="mean", dtype=torch.float32):
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=dtype)
    labels = labels.view(-1, 1)
    prediction_logprobs = torch.gather(logprobs, dim=-1, index=labels)
    prediction_logprobs = prediction_logprobs.squeeze(-1)

    if reduction == "mean":
        loss = -torch.mean(prediction_logprobs)
    elif reduction == "sum":
        loss = -torch.sum(prediction_logprobs)
    elif reduction == "none":
        loss = -prediction_logprobs
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def update_results(filename, experiment_key, logger_metrics):
    try:
        results = torch.load(filename)
    except:
        results = {}
        
    results[experiment_key] = logger_metrics
    torch.save(results, filename)

def evaluate(model, data_loader, dtype=torch.float64):
    model.eval()
    loss = 0
    correct = 0
    device = next(model.parameters()).device
    float_precision = next(model.parameters()).dtype
    with torch.no_grad():
        for data, target in data_loader:
            label_argmax = len(target.shape)!=1
            data = data.to(device)
            data = data.to(float_precision)
            output = model(data).to("cpu")

            loss += softmax_cross_entropy(output, target, dtype=dtype).item()
            pred = output.argmax(dim=1, keepdim=True)
            if label_argmax:
                target = target.argmax(dim=1)
            correct += pred.eq(target.to("cpu").view_as(pred)).sum().item()
    loss /= len(data_loader)
    accuracy = 100 * correct / len(data_loader.dataset)
    return loss, accuracy

def evaluate_xor(model, data_loader, amp=0.0, dtype=torch.float64):
    model.eval()
    correct = 0
    device = next(model.parameters()).device
    float_precision = next(model.parameters()).dtype
    with torch.no_grad():
        for X, y in data_loader:
            X,y = X.to(device),y.to(device)
            X = X.to(float_precision)
            X[:, 2:] += torch.randn_like(X[:, 2:]) * amp
            preds = torch.sign(model(X).squeeze())
            correct += (preds == y).float().sum().item()

    accuracy = 100 * correct / len(data_loader.dataset)
    return accuracy

def get_specified_args(parser, args):

    defaults = {action.dest: action.default
                for action in parser._actions
                if action.dest != 'help'}
    
    specified = {arg: getattr(args, arg)
                 for arg in vars(args)
                 if getattr(args, arg) != defaults.get(arg)
                 and arg!="device"}
    
    return specified

def split_dataset(dataset, train_fraction):
    total_size = len(dataset)
    train_size = int(train_fraction * total_size)
    test_size = total_size - train_size
    print(f'Starting training. Train dataset size: {train_size}, Test size: {test_size}')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def reduce_train_dataset(original_train_dataset, reduced_fraction, batch_size):
    original_indices = original_train_dataset.indices
    reduced_train_size = int(reduced_fraction * len(original_indices))
    reduced_indices = original_indices[:reduced_train_size]
    reduced_train_dataset = Subset(original_train_dataset, reduced_indices)
    
    reduced_train_loader = DataLoader(reduced_train_dataset, batch_size=batch_size, shuffle=True)
    return reduced_train_loader

def generate_xor_dataset(n, p, epsilon):
    x_signal = np.random.choice([-1, 1], size=(n, 2))
    x_noise = np.random.choice([-epsilon, epsilon], size=(n, p - 2))
    X = np.hstack([x_signal, x_noise])
    y = x_signal[:, 0] * x_signal[:, 1]
    X=torch.tensor(X)
    y=torch.tensor(y)
    return torch.utils.data.TensorDataset(X,y)

def get_dataset(args):
    if args.dataset=='sparse_parity':
        dataset = SparseParityDataset(args.num_parity_features, args.num_noise_features, args.num_samples)
        train_dataset, test_dataset = split_dataset(dataset, args.train_fraction)
    elif args.dataset=='mnist':
        MNIST_TRANSFORM=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,)),torchvision.transforms.Lambda(lambda x: x.flatten())])
        train_dataset = torchvision.datasets.MNIST(root=args.dir, train=True, transform=MNIST_TRANSFORM, download=True)
        test_dataset = torchvision.datasets.MNIST(root=args.dir, train=False, transform=MNIST_TRANSFORM, download=True)
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, range(args.num_samples))
    elif args.dataset=='xor':
        train_dataset=generate_xor_dataset(args.n,args.p,args.epsilon)
        test_dataset=generate_xor_dataset(args.n,args.p,args.epsilon)

    return train_dataset, test_dataset

def generate_random_one_hot(length):
    index = torch.randint(0, length, (1,)).item()
    one_hot_vector = torch.zeros(length)
    one_hot_vector[index] = 1
    return one_hot_vector

def get_model(args):
    activation_func = getattr(torch.nn, args.activation_function)
    bias=True
    if args.dataset=='sparse_parity':
        input_size=args.num_parity_features + args.num_noise_features
        output_size=2
    elif args.dataset=='mnist':
        input_size=784
        output_size=10
    elif args.dataset=='xor':
        input_size=args.p
        output_size=1
        bias=False
    model = MLP(input_size=input_size, output_size=output_size,hidden_sizes=args.hidden_sizes, non_linearity=activation_func, bias=bias)

    return model
        
def get_optimizer(model, args):

    param_group = dict(params=model.parameters())

    if args.optimizer in ("Adam", "AdamW"):
        param_group |= dict(lr=args.lr, betas=(0.9, args.beta2), eps=args.adam_epsilon)
    elif args.optimizer == "SGD":
        param_group |= dict(lr=args.lr, momentum=0.8 if args.orthogonal_gradients else 0.2)
    else:
        raise ValueError(f'Unsupported optimizer type: {args.optimizer}')

    param_group |= dict(weight_decay=args.weight_decay)
    base_optimizer_cls = getattr(optim, args.optimizer)

    if args.orthogonal_gradients:
        optimizer = OrthoGrad([param_group], base_optimizer_cls)
    else:
        optimizer = base_optimizer_cls([param_group])

    return optimizer


######### MOD ADD GROKKING 

class OneLayerFCN(torch.nn.Module):
  def __init__(self, num_tokens: int, hidden_width: int,
               context_len: int, init_scale=1.0, n_classes=-1):
    super().__init__()

    if n_classes == -1:
        n_classes = num_tokens

    self.num_tokens = num_tokens
    inp_dim = self.num_tokens * context_len
    self.inp_dim = inp_dim
    self.hidden_width = hidden_width
    self.init_scale = init_scale

    self.fc1 = nn.Linear(inp_dim, hidden_width, bias=False)
    self.out = nn.Linear(hidden_width, n_classes, bias=False)

    self.reset_params(init_scale=init_scale)

  def reset_params(self, init_scale=1.0):
    # scaled kaiming uniform code:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(self.fc1.weight, -init_scale*bound, init_scale*bound)

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(self.out.weight, -init_scale*bound, init_scale*bound)

    # scaled kaiming normal code:
    '''
    leaky_neg_slope = 0.
    fan = nn.init._calculate_correct_fan(self.fc1.weight, "fan_in")
    gain = nn.init.calculate_gain("leaky_relu", leaky_neg_slope)
    std = gain/math.sqrt(fan)
    nn.init.normal_(self.fc1.weight, mean=0.0, std=init_scale*std)

    fan = nn.init._calculate_correct_fan(self.out.weight, "fan_in")
    gain = nn.init.calculate_gain("leaky_relu", leaky_neg_slope)
    std = gain/math.sqrt(fan)
    nn.init.normal_(self.out.weight, mean=0.0, std=init_scale*std)
    '''

  def forward(self, x, dumb1=None, act='relu'):
      if act == 'relu':
          act_fn = F.relu
      elif act == 'swish':
          act_fn = F.silu
      elif act == 'quadratic':
          act_fn = lambda x: torch.pow(x, 2)
      elif act == 'softplus':
          act_fn = F.softplus
      elif act == 'linear':
          act_fn = lambda x: x
      elif act == 'hermite2':
          act_fn = lambda x: (torch.pow(x, 2) - 1)/math.sqrt(2)

      if dumb1 is None:
          x = self.fc1(x)
          x = act_fn(x)

          return self.out(x)

      x = act_fn(self.fc1(x) + dumb1 @ self.fc1.weight.t())
      x = self.out(x)
      return x
  
def evaluate_mod_add(model,loader,args):
    model.eval()
    with torch.no_grad():
        count = 0
        total = 0
        for idx, batch in enumerate(loader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = batch

            output = model(inputs, act=args.act_fn)

            count += (output.argmax(-1) == labels.argmax(-1)).sum().item()
            total += output.shape[0]

        acc = (count / total) * 100
    return acc
  
# based on:
# https://github.com/danielmamay/grokking/blob/main/grokking/data.py

from math import ceil
import torch
import itertools
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.float64)

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
    "(x//y)if(y%2==1)else(x-y)": lambda x, y, _: torch.where(y % 2 == 1, x // y, x - y)
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    "x*y": lambda x, y, _: (x, y, x*y),
    **DIVISION_MODULO_OPERATIONS,
    "x^2+y": lambda x, y, _: (x, y, x**2 + y),
    "x^2+y^2": lambda x, y, _: (x, y, x**2 + y**2),
    "x^2+xy+y^2": lambda x, y, _: (x, y, x**2 + x*y + y**2),
    "x^2+xy+y^2+x": lambda x, y, _: (x, y, x**2 + x*y + y**2 + x),
    "x^3+xy": lambda x, y, _: (x, y, x**3 + x*y),
    "x^3+xy^2+x": lambda x, y, _: (x, y, x**3 + x*y**2 + y)
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    x, y, z = ALL_OPERATIONS[operation](x, y, p)
    results = z.remainder(p)

    inputs = torch.stack([x, y], dim=1)
    labels = results

    return inputs, labels

def multitask_op_mod_p_data(op1, op2, p, train_frac_per_op):
    inp1, lab1 = operation_mod_p_data(op1, p)
    X_tr1, y_tr1, X_te1, y_te1 = make_data_splits(inp1, lab1, train_frac_per_op)
    X_tr1 = F.one_hot(X_tr1, p).view(-1, 2*p).double()
    X_te1 = F.one_hot(X_te1, p).view(-1, 2*p).double()
    X_tr2 = X_tr1.clone()
    X_te2 = X_te1.clone()

    task2_dim = p

    zeros = torch.zeros((X_tr1.shape[0],1))
    X_tr1 = torch.hstack((X_tr1, zeros))
    y_tr1 = F.one_hot(y_tr1, p).double()

    zeros = torch.zeros((X_te1.shape[0],1))
    X_te1 = torch.hstack((X_te1, zeros))
    y_te1 = F.one_hot(y_te1, p).double()

    ones = torch.ones((X_tr2.shape[0], 1))
    dig1 = X_tr2[:,:p].argmax(-1)
    dig2 = X_tr2[:,p:].argmax(-1)

    _, _, y_tr2 = ALL_OPERATIONS[op2](dig1, dig2, p)
    y_tr2 = y_tr2.remainder(p)

    y_tr2 = F.one_hot(y_tr2, task2_dim).double()
    X_tr2 = torch.hstack((X_tr2, ones))

    ones = torch.ones((X_te2.shape[0], 1))
    dig1 = X_te2[:,:p].argmax(-1)
    dig2 = X_te2[:,p:].argmax(-1)
    _, _, y_te2 = ALL_OPERATIONS[op2](dig1, dig2, p)
    y_te2 = y_te2.remainder(p)

    y_te2 = F.one_hot(y_te2, task2_dim).double()
    X_te2 = torch.hstack((X_te2, ones))

    X_tr = torch.vstack((X_tr1, X_tr2))
    y_tr = torch.vstack((y_tr1, y_tr2))

    return X_tr, y_tr, X_te1, y_te1, X_te2, y_te2

def make_data_splits(inputs, labels, training_fraction):
    train_size = int(training_fraction * inputs.shape[0])
    val_size = inputs.shape[0] - train_size

    perm = torch.randperm(inputs.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]

def make_dataloader(inputs, labels, batch_size, shuffle=False, drop_last=False):
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    batch_size = min(batch_size, ceil(len(dataset) / 2))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)