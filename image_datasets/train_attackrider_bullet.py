import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision import transforms

from models.resnet import *
from models.resnet_tinyimagenet import ResNet18_tinyimagenet


from utils import *
from TinyImageNet_utils.TinyImageNet import TinyImageNet_idx

DATASET_PATH = '../../CIFAR-10_AT/cifar-data'

def bullet_pgd(model,
                x_natural,
                y,
                acc_ref,
                err_ref,
                p_percent,
                optimizer,
                step_size,
                epsilon,
                perturb_steps,
                beta=6.0,
                perturb_steps_prime=2, 
                step_size_prime = 6./255):
    criterion_ce = nn.CrossEntropyLoss(reduction='sum')
    x_adv = x_natural.clone().detach()
    model.eval()
    subfile_size = len(x_natural)
    with torch.no_grad():
        natural_output = model(x_natural)
        natural_loss = F.softmax(natural_output, dim=1)

    gamma = 0.8

    batch_err = 1.0 * (natural_output.max(1)[1] != y).detach().float().mean() 

    sign = 2 * ((natural_output.max(1)[1] == y).float()) - 1
    pred_var = (torch.std(natural_loss, dim=1) * sign).detach()

    p = torch.clamp(1.0-acc_ref, 0, 1) # in [0, 1]
    th = F.relu(torch.quantile(pred_var, p))

    pred_robust = (pred_var > th).int()
    pred_boundary = torch.logical_and(pred_var <= th, pred_var >= 0).int()

    r_size = pred_robust.sum()
    b_size = pred_boundary.sum()

    robust_index = torch.nonzero(pred_robust)[:, 0]
    boundary_index = torch.nonzero(pred_boundary)[:, 0]
    
    if b_size > 0:
        x2, y2 = x_natural[boundary_index, :, :, :], y[boundary_index]
        x_adv2 = x2.detach() + 0.001 * torch.randn(x2.shape).cuda().detach()
        a2 = torch.ones_like(y2) * step_size
        a2 = a2.reshape(a2.shape[0], 1, 1, 1)
        for _ in range(perturb_steps): 
            x_adv2.requires_grad_()
            with torch.enable_grad():
                output = model(x_adv2)
                loss_kl = criterion_ce(output, y2)
            grad = torch.autograd.grad(loss_kl, [x_adv2])[0]
            x_adv2 = x_adv2.detach() + a2 * torch.sign(grad.detach())
            x_adv2 = torch.min(torch.max(x_adv2, x2 - epsilon), x2 + epsilon)
            x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)

        x_adv[boundary_index] = x_adv2

    if r_size > 0:
        x2, y2 = x_natural[robust_index, :, :, :], y[robust_index]
        x_adv2 = x2.detach() + 0.001 * torch.randn(x2.shape).cuda().detach()
        a2 = torch.ones_like(y2) * step_size_prime
        a2 = a2.reshape(a2.shape[0], 1, 1, 1)
        for _ in range(perturb_steps_prime):  
            x_adv2.requires_grad_()
            with torch.enable_grad():
                output = model(x_adv2)
                loss_kl = criterion_ce(output, y2)
            grad = torch.autograd.grad(loss_kl, [x_adv2])[0]
            x_adv2 = x_adv2.detach() + a2 * torch.sign(grad.detach())
            x_adv2 = torch.min(torch.max(x_adv2, x2 - epsilon), x2 + epsilon)
            x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)

        x_adv[robust_index] = x_adv2
    
    x_adv.requires_grad_(False)
    x_natural.requires_grad_(False)
    
    with torch.no_grad():
        output_adv_subfile = model(x_adv)
        batch_acc = gamma * (output_adv_subfile.max(1)[1] == y).detach().float().mean()

    model.train()
    x_adv.requires_grad_(False)
    x_natural.requires_grad_(False)
    
    # Shuffle
    shuffle_index = torch.randperm(subfile_size)
    x_adv = x_adv[shuffle_index]
    y = y[shuffle_index]
    
    # Divide and Model Update
    for batch_start in range(0, subfile_size, args.batch_size):
        y_batch     = y[batch_start:batch_start+args.batch_size] 
        x_batch_AEs = x_adv[batch_start:batch_start+args.batch_size]
        
        optimizer.zero_grad()
        output_adv = model(x_batch_AEs)
        loss = F.cross_entropy(output_adv, y_batch)
        loss.backward()
        optimizer.step()

    

    
    with torch.no_grad():
        F_O = batch_err
        F_B = b_size / subfile_size
        F_R = r_size / subfile_size 
    
    return loss, (batch_acc, batch_err), (F_O, F_B, F_R)

def bullet_trades(model,
                x_natural,
                y,
                acc_ref,
                err_ref,
                p_percent,
                optimizer,
                step_size,
                epsilon,
                perturb_steps,
                beta=6.0,
                perturb_steps_prime=2, 
                step_size_prime = 6./255):
    criterion_kl = nn.KLDivLoss(reduction='sum')
    jitter = 1e-12
    x_adv = x_natural.clone().detach()
    model.eval()
    subfile_size = len(x_natural)

    with torch.no_grad():
        natural_output = model(x_natural)
        natural_loss = F.softmax(natural_output, dim=1)
    
    gamma = 0.8

    batch_err = 1.0 * (natural_output.max(1)[1] != y).detach().float().mean() 

    sign = 2 * ((natural_output.max(1)[1] == y).float()) - 1
    pred_var = (torch.std(natural_loss, dim=1) * sign).detach()

    p = torch.clamp(1.0-acc_ref, 0, 1) 
    th = F.relu(torch.quantile(pred_var, p))

    pred_robust = (pred_var > th).int()
    pred_boundary = torch.logical_and(pred_var <= th, pred_var >= 0).int()

    r_size = pred_robust.sum()
    b_size = pred_boundary.sum()

    robust_index = torch.nonzero(pred_robust)[:, 0]
    boundary_index = torch.nonzero(pred_boundary)[:, 0]
    
    if b_size > 0:
        x2, y2 = x_natural[boundary_index, :, :, :], y[boundary_index]
        x_adv2 = x2.detach() + 0.001 * torch.randn(x2.shape).cuda().detach()
        natural_loss2 = natural_loss[boundary_index]
        a2 = torch.ones_like(y2) * step_size
        a2 = a2.reshape(a2.shape[0], 1, 1, 1)
        for _ in range(perturb_steps):
            x_adv2.requires_grad_()
            with torch.enable_grad():
                output = model(x_adv2)
                loss_kl = criterion_kl(F.log_softmax(output + jitter, dim=1), natural_loss2 + jitter)
            grad = torch.autograd.grad(loss_kl, [x_adv2])[0]
            x_adv2 = x_adv2.detach() + a2 * torch.sign(grad.detach())
            x_adv2 = torch.min(torch.max(x_adv2, x2 - epsilon), x2 + epsilon)
            x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)

        x_adv[boundary_index] = x_adv2

    if r_size > 0:
        x2, y2 = x_natural[robust_index, :, :, :], y[robust_index]
        x_adv2 = x2.detach() + 0.001 * torch.randn(x2.shape).cuda().detach()
        natural_loss2 = natural_loss[robust_index]
        a2 = torch.ones_like(y2) * step_size_prime
        a2 = a2.reshape(a2.shape[0], 1, 1, 1)
        for _ in range(perturb_steps_prime):
            x_adv2.requires_grad_()
            with torch.enable_grad():
                output = model(x_adv2)
                loss_kl = criterion_kl(F.log_softmax(output + jitter, dim=1), natural_loss2 + jitter)
            grad = torch.autograd.grad(loss_kl, [x_adv2])[0]
            x_adv2 = x_adv2.detach() + a2 * torch.sign(grad.detach())
            x_adv2 = torch.min(torch.max(x_adv2, x2 - epsilon), x2 + epsilon)
            x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)

        x_adv[robust_index] = x_adv2
    
    x_adv.requires_grad_(False)
    x_natural.requires_grad_(False)
    
    with torch.no_grad():
        output_adv_subfile = model(x_adv)
        batch_acc = gamma * (output_adv_subfile.max(1)[1] == y).detach().float().mean()

    model.train()
    x_adv.requires_grad_(False)
    x_natural.requires_grad_(False)
    
    # Shuffle
    shuffle_index = torch.randperm(subfile_size)
    x_natural = x_natural[shuffle_index]
    x_adv = x_adv[shuffle_index]
    y = y[shuffle_index]
    
    # Divide and Model Update
    for batch_start in range(0, subfile_size, args.batch_size): 
        x_batch     = x_natural[batch_start:batch_start+args.batch_size]
        y_batch     = y[batch_start:batch_start+args.batch_size] 
        x_batch_AEs = x_adv[batch_start:batch_start+args.batch_size]
        
        actual_batch_size = len(x_batch)
        optimizer.zero_grad()
        
        output_adv = model(x_batch_AEs)
        logits = model(x_batch)
        loss_natural = F.cross_entropy(logits, y_batch)
        loss_robust = (1.0 / actual_batch_size) * criterion_kl(F.log_softmax(output_adv + jitter, dim=1), F.softmax(logits, dim=1) + jitter)
        loss = loss_natural + beta * loss_robust
        
        loss.backward()
        optimizer.step()

    
    with torch.no_grad():
        F_O = batch_err
        F_B = b_size / subfile_size
        F_R = r_size / subfile_size
    
    return loss, (batch_acc, batch_err), (F_O, F_B, F_R)
    



parser = argparse.ArgumentParser()

parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--base-at', default="Bullet_TRADES", type=str) # {Bullet_PGD, Bullet_TRADES}
parser.add_argument('--dataset', default="CIFAR-10", type=str) # {CIFAR-10, CIFAR-100, TinyImageNet}
parser.add_argument('--e', default=1, type=int) 
parser.add_argument('--beta', default=10.0, type=float)
parser.add_argument('--model-dir', default=None, type=str)
parser.add_argument('--seed', default=1, type=int)

args = parser.parse_args()

cudnn.benchmark = False
cudnn.deterministic = True
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if args.model_dir is None:
    args.model_dir = 'results/AR_{}_beta{}_e{}'.format(args.base_at, args.beta, args.e) 

args.model_dir = args.dataset + "_" + args.model_dir

args.base_model = "RN"

args.batch_size = 128
args.test_batch_size = 1000
args.epochs = 120 
args.lr = 0.1
args.momentum = 0.9
args.num_steps = 10
args.log_interval = 1
args.eval_attack_batches = 50
args.num_workers = 4


args.epsilon = 8./255
args.step_size = 2./255
args.test_epsilon = 8./255
args.test_step_size = 2./255
args.test_num_steps= 20
args.weight_decay = 2e-4 if args.dataset == "CIFAR-10" else 5e-4
args.lr_mode = "120e"

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir): os.makedirs(model_dir)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(model_dir, "training.log")),
        logging.StreamHandler(),
    ],
)
logging.info("Input args: %r", args)

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == "CIFAR-10":
    trainset = CIFAR10_idx(root=DATASET_PATH, train=True, download=False, transform=transform_train)
    testset = CIFAR10_idx(root=DATASET_PATH, train=False, download=False, transform=transform_test)
    eval_trainset = trainset
    
    train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    eval_train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == "CIFAR-100":
    trainset = CIFAR100_idx(root=DATASET_PATH, train=True, download=False, transform=transform_train)
    testset = CIFAR100_idx(root=DATASET_PATH, train=False, download=False, transform=transform_test)
    eval_trainset = trainset
    
    train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    eval_train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == "TinyImageNet":
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    trainset = TinyImageNet_idx('.', 'train', transform=transform_train, in_memory=True)
    testset = TinyImageNet_idx('.', 'val', transform=transform_test, in_memory=True)
    eval_trainset = trainset
    
    train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    eval_train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch, acc_ref, err_ref): 
    model.train()
    F_Bs = []
    loss_epoch = 0
    ns = []
    batch_count = 0

    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_count += 1
        p_percent = 1 - acc_ref - err_ref
        p_percent = torch.clamp(p_percent, 0.1, 1)
        

        if args.e == -1:
            n = 1
            x_natural = data
            y = target
        else:
            n = int(torch.ceil(args.m_prime / (p_percent * args.batch_size) ).item())
            if batch_count % n == 1: # the first batch of the superbatch
                x_natural = data
                y = target
                continue
            else:  # the following batch of the superbatch
                x_natural = torch.cat((x_natural, data), 0)
                y = torch.cat((y, target), 0)
                if batch_count % n != 0 and (batch_idx+1) != len(train_loader):
                    continue
        
        batch_count = 0
        ns.append(n)
        optimizer.zero_grad()
        if "TRADES" in args.base_at:
            loss, (batch_acc, batch_err), (F_O, F_B, F_R) = bullet_trades(model=model,
                                                    x_natural=x_natural,
                                                    y=y,
                                                    acc_ref=acc_ref,
                                                    err_ref=err_ref,
                                                    p_percent=p_percent,
                                                    optimizer=optimizer,
                                                    step_size=args.step_size,
                                                    epsilon=args.epsilon,
                                                    perturb_steps=args.num_steps,
                                                    beta=args.beta)
        elif "PGD" in args.base_at:
            loss, (batch_acc, batch_err), (F_O, F_B, F_R) = bullet_pgd(model=model,
                                                    x_natural=x_natural,
                                                    y=y,
                                                    acc_ref=acc_ref,
                                                    err_ref=err_ref,
                                                    p_percent=p_percent,
                                                    optimizer=optimizer,
                                                    step_size=args.step_size,
                                                    epsilon=args.epsilon,
                                                    perturb_steps=args.num_steps)
        
        loss_epoch += loss.item()
        acc_ref = acc_ref * 0.9 + batch_acc * 0.1
        err_ref = err_ref * 0.9 + batch_err * 0.1

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF_B: {:.4f}\tF_R: {:.4f}\tF_O: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), 
                    F_B, F_R, F_O))
            
            
        F_Bs.append(F_B)
    
    avg_F_B = sum(F_Bs) / len(F_Bs)
    
    return avg_F_B, acc_ref, err_ref, ns
        



def main():
    assert args.base_model == "RN"
    assert args.dataset in ["CIFAR-10", "CIFAR-100", "TinyImageNet"]
    
    args.m_prime = args.e * args.batch_size
    
    if args.dataset == "TinyImageNet":
        model = ResNet18_tinyimagenet().to(device)
    else: 
        model = ResNet18(num_classes=10 if args.dataset == "CIFAR-10" else 100).to(device)
    
    model = nn.DataParallel(model)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:  
        state_dicts = torch.load(args.resume)
        model.load_state_dict(state_dicts["state_dict"])
        optimizer.load_state_dict(state_dicts["opt_state_dict"])
        start_epoch = 1 + state_dicts["epoch"]
    else: 
        start_epoch = 1
    
    
    best_robust_acc = 0
    best_clean_acc = 0
    best_epoch = -1
    epoch_times = []
    acc_ref = torch.zeros(1).cuda()
    err_ref = torch.ones(1).cuda() * 0.9
    F_Bs = []
    ns = []
    for epoch in range(start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(args.lr, optimizer, epoch, mode=args.lr_mode)

        start_time = time.time()
        F_B_epoch, acc_ref, err_ref, ns_epoch  = train(args, model, device, train_loader, optimizer, epoch, acc_ref, err_ref)
        epoch_time = time.time() - start_time
        logging.info("Epoch Time: {} s".format(epoch_time))
        epoch_times.append(epoch_time)
        ns += ns_epoch
        F_Bs.append(F_B_epoch)
        
        # save checkpoint
        save_dict = {"state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "robust_acc": -1,
                    "clean_acc": -1}
        torch.save(save_dict, 
                os.path.join(model_dir, 'ep_cur.pt'))
        
        # if epoch < 78: continue
        logging.info('================================================================')
        
        start_time = time.time()
        clean_accuracy, robust_accuracy = evaluate(args, model, device, 'test', test_loader, eval_attack_batches = args.eval_attack_batches)

        logging.info("Evaluation Time: {} s".format(time.time() - start_time))
        
        logging.info('================================================================')
        

        if robust_accuracy >= best_robust_acc:
            torch.save(save_dict, 
                    os.path.join(model_dir, 'ep_best.pt'))
            best_robust_acc = robust_accuracy
            best_clean_acc = clean_accuracy
            best_epoch = epoch
        
        
    logging.info("Best epoch: {}, RA: {:.2f}%, SA: {:.2f}%".format(best_epoch, 100.0*best_robust_acc, 100.0*best_clean_acc))
    logging.info("Avg Epoch Time and std: {:.4f}, {:.4f}".format(np.mean(epoch_times), np.std(epoch_times)))
    logging.info("Total Time: {:.4f}".format(np.sum(epoch_times)))
    logging.info("F_Bs: {}".format(F_Bs))
    logging.info("avg_n: {}".format(np.mean(ns)))

    
if __name__ == '__main__':
    main()
