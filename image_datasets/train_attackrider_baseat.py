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


parser = argparse.ArgumentParser()

parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--base-at', type=str)
parser.add_argument('--model-dir', default=None, type=str)
parser.add_argument('--beta', default=6.0, type=float)
parser.add_argument('--dataset', default='CIFAR-10', type=str) 
parser.add_argument('--e', default=1, type=int)
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

args.samples_per_class_train = 1000
args.samples_per_class_test = 1000

args.epsilon = 8./255
args.step_size = 2./255
args.test_epsilon = 8./255
args.test_step_size = 2./255
args.test_num_steps= 20
args.weight_decay = 2e-4 if args.dataset == "CIFAR-10" else 5e-4


args.lr_mode = "120e"


model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
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

        
def train_atkrider(args, model, device, train_loader, optimizer, epoch): 
    model.train()
    batch_count = 0
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_count += 1

        if batch_count % args.e == 1: # first batch 
            x_natural = data
            y = target
            continue
        elif args.e == 1:  # normal single-batch training
            x_natural = data
            y = target
        else:  # second to e-th batch
            x_natural = torch.cat((x_natural, data), 0)
            y = torch.cat((y, target), 0)
            if batch_count % args.e != 0 and (batch_idx+1) != len(train_loader):
                continue
        
        batch_count = 0
        optimizer.zero_grad()
        if args.base_at == "PGDAT":
            model.eval()
            criterion = nn.CrossEntropyLoss(reduction='sum')
            # generate adversarial example
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            for _ in range(args.num_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_AE = criterion(model(x_adv), y)
                grad = torch.autograd.grad(loss_AE, [x_adv])[0]
                x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            model.train()

            x_adv.requires_grad_(False)

        elif args.base_at == "TRADES":
            criterion_kl = nn.KLDivLoss(reduction='sum')
            model.eval()
            jitter = 1e-12
            p_natural = F.softmax(model(x_natural), dim=1).detach()
            # generate adversarial example
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            for _ in range(args.num_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv) + jitter, dim=1),
                                            p_natural + jitter)
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
            model.train()

            x_adv.requires_grad_(False)
        else:
            raise NotImplementedError
        
        # model update
        subfile_size = x_natural.shape[0]
        for batch_start in range(0, subfile_size, args.batch_size):
            X_batch = x_natural[batch_start:batch_start+args.batch_size]
            X_adv_batch = x_adv[batch_start:batch_start+args.batch_size]
            y_batch = y[batch_start:batch_start+args.batch_size]
            
            true_batch_size = X_batch.size(0)
            if args.base_at == "PGDAT":
                logits_adv = model(X_adv_batch)
                loss = F.cross_entropy(logits_adv, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif args.base_at == 'TRADES':
                
                output_adv = model(X_adv_batch)
                logits = model(X_batch)
                loss_natural = F.cross_entropy(logits, y_batch)
                loss_robust = (1.0 / true_batch_size) * criterion_kl(F.log_softmax(output_adv + jitter, dim=1), F.softmax(logits, dim=1) + jitter)
                loss = loss_natural + args.beta * loss_robust
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def main():
    assert args.base_model == "RN"
    assert args.dataset in ["CIFAR-10", "CIFAR-100", "TinyImageNet"]
    
    if args.dataset == "TinyImageNet":
        model = ResNet18_tinyimagenet().to(device)
    else:
        model = ResNet18(num_classes=10 if args.dataset != "CIFAR-100" else 100).to(device)

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
    
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args.lr, optimizer, epoch, mode=args.lr_mode)

        start_time = time.time()
        train_atkrider(args, model, device, train_loader, optimizer, epoch)
        epoch_time = time.time() - start_time
        logging.info("Epoch Time: {} s".format(epoch_time))
        epoch_times.append(epoch_time)
        
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
