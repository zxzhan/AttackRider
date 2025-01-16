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

from typing import List
import math

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--base-model', default='RN', type=str) 
parser.add_argument('--model-dir', default=None, type=str)
parser.add_argument('--beta', default=6.0, type=float)
parser.add_argument('--e', type=int) 
parser.add_argument('--dataset', default='CIFAR-10', type=str)

args = parser.parse_args()

cudnn.benchmark = False
cudnn.deterministic = True
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


if not args.model_dir: 
    args.model_dir = 'results/AR_DBAC' + '_e' + str(args.e) 

args.model_dir = args.dataset + "_" + args.model_dir

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
        



def main():
    assert args.base_model == "RN"
    assert args.dataset in ["CIFAR-10", "CIFAR-100", "TinyImageNet"]

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
    
    def get_FB_epoch(epochs): 
        all_start_epoch = epochs * 0.9
        F_B = [0.5 / all_start_epoch * epoch + 0.5 if epoch<=all_start_epoch else 1 for epoch in range(epochs)] 
        F_B = [-1] + F_B
        return F_B

    F_B = get_FB_epoch(args.epochs)
    
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args.lr, optimizer, epoch, mode=args.lr_mode)
        b_epoch = math.ceil(args.e / (F_B[epoch])) if args.e > 0 else -args.e
        boundary_size = math.ceil(F_B[epoch] * b_epoch * args.batch_size)
        print("Epoch: {}: b: {}, F_B: {}, Boundary Size: {}".format(epoch, b_epoch, F_B[epoch], boundary_size))
        start_time = time.time()

        model.train()
        
        # =================================== Training Start ===================================
        batch_count = 0
        for batch_idx, (data, target, indices) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            batch_count += 1
            # print(data.shape, target.shape, indices.shape)

            if batch_count % b_epoch == 1: # the first batch of the superbatch
                x_natural = data
                y = target
                continue
            elif b_epoch == 1:  # normal training
                x_natural = data
                y = target
            else:  # the following batch of the superbatch
                x_natural = torch.cat((x_natural, data), 0)
                y = torch.cat((y, target), 0)
                if batch_count % b_epoch != 0 and (batch_idx+1) != len(train_loader):
                    continue
            
            batch_count = 0
            
            # ================================== PGD's Loss ==================================
            criterion_ce = nn.CrossEntropyLoss(reduction='sum')
            x_adv = x_natural.clone().detach()
            model.eval()
            subfile_size = len(x_natural)
            random_index = torch.randperm(subfile_size)
            boundary_index = random_index[:boundary_size]
            remaining_index = random_index[boundary_size:]
            

            x2 = x_natural[boundary_index]
            y2 = y[boundary_index]
            
            x2, y2 = x_natural[boundary_index, :, :, :], y[boundary_index]
            x_adv2 = x2.detach() + 0.001 * torch.randn(x2.shape).cuda().detach()
            a2 = torch.ones_like(y2) * args.step_size
            a2 = a2.reshape(a2.shape[0], 1, 1, 1)
            for _ in range(args.num_steps):  
                x_adv2.requires_grad_()
                with torch.enable_grad():
                    output = model(x_adv2)
                    loss_kl = criterion_ce(output, y2)
                grad = torch.autograd.grad(loss_kl, [x_adv2])[0]
                x_adv2 = x_adv2.detach() + a2 * torch.sign(grad.detach())
                x_adv2 = torch.min(torch.max(x_adv2, x2 - args.epsilon), x2 + args.epsilon)
                x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)

            x_adv[boundary_index] = x_adv2
            
            
            x_adv.requires_grad_(False)
            model.train()

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
            
            # Logging
            if batch_idx % args.log_interval == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        
        epoch_time = time.time() - start_time
        logging.info("Epoch Time: {} s".format(epoch_time))
        epoch_times.append(epoch_time)
        
        # =================================== Training end ===================================
         
        logging.info('================================================================')
        
        start_time = time.time()
        clean_accuracy, robust_accuracy = evaluate(args, model, device, 'test', test_loader, eval_attack_batches = args.eval_attack_batches)
        
        logging.info("Evaluation Time: {} s".format(time.time() - start_time))
        
        logging.info('================================================================')
        
        # save checkpoint
        save_dict = {"state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "robust_acc": robust_accuracy,
                    "clean_acc": clean_accuracy}
        torch.save(save_dict, 
                os.path.join(model_dir, 'ep_cur.pt'))
        if robust_accuracy >= best_robust_acc:
            torch.save(save_dict, 
                    os.path.join(model_dir, 'ep_best.pt'))
            best_robust_acc = robust_accuracy
            best_clean_acc = clean_accuracy
            best_epoch = epoch
        
    logging.info("Best epoch: {}, RA: {:.2f}%, SA: {:.2f}%".format(best_epoch, 100.0*best_robust_acc, 100.0*best_clean_acc))
    logging.info("Avg Epoch Time and std: {:.4f}, {:.4f}".format(np.mean(epoch_times), np.std(epoch_times)))
    logging.info("Total Time: {:.4f}".format(np.sum(epoch_times)))

        
if __name__ == '__main__':
    main()
