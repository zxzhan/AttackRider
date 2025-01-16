import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time

DATASET_PATH = '../../CIFAR-10_AT/cifar-data'


def adjust_learning_rate(base_lr, optimizer, epoch, mode=None):
    """decrease the learning rate"""
    lr = base_lr
    if mode == "120e":
        if epoch >= 80:
            lr = base_lr * 0.1
        if epoch >= 100:
            lr = base_lr * 0.01
    elif mode == "80e":
        if epoch >= 50:
            lr = base_lr * 0.1
        if epoch >= 65:
            lr = base_lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pgd(model,
        X,
        y,
        epsilon,
        num_steps,
        step_size):
    model.eval()
    X_ori = X.clone()
    X = X + X.new(X.size()).uniform_(-epsilon, epsilon)
    for _ in range(num_steps):
        X.requires_grad_(True)
        out = model(X)
        loss = F.cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        X = X.data + step_size * X.grad.data.sign()
        delta_X = torch.clamp(X - X_ori, -epsilon, epsilon)
        X = torch.clamp(X_ori + delta_X, 0, 1)
    model.zero_grad()
    with torch.no_grad():
        out = model(X)
        pred = out.data.argmax(1)
        n_correct_adv = (pred == y).sum()
    return n_correct_adv

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size,
                epsilon,
                perturb_steps,
                beta):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    jitter = 1e-12
    p_natural = F.softmax(model(x_natural), dim=1).detach()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv) + jitter, dim=1),
                                    p_natural + jitter)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()

    x_adv.requires_grad_(False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv + jitter, dim=1),
                                                    F.softmax(logits_natural, dim=1) + jitter)
    loss = loss_natural + beta * loss_robust
    
    return loss




class CIFAR10_idx(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10_idx, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(CIFAR10_idx, self).__getitem__(index)
        return img, target, index    

class CIFAR100_idx(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CIFAR100_idx, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(CIFAR100_idx, self).__getitem__(index)
        return img, target, index    


def update_batch_allocations(percentage, batch_allocation_sampler, margins, attack_indication):
    start_point = 0.0
    
    # Attack samples selection, (non_)attack_indices is w.r.t x_natural.
    # -------------------------------
    non_neg_indices = (margins >= start_point).nonzero().squeeze()
    attack_margins = margins[non_neg_indices]
    attack_num = int(percentage * len(margins))
    sorted_indices = torch.argsort(attack_margins)
    attack_indices = sorted_indices[:attack_num]
    non_attack_indices = sorted_indices[attack_num:]
    attack_indices = non_neg_indices[attack_indices]
    non_attack_indices = non_neg_indices[non_attack_indices]
    non_attack_indices = torch.cat([non_attack_indices, (margins < start_point).nonzero().squeeze()], dim=0)
    # -------------------------------

    attack_indication[attack_indices] = True
    attack_indication[non_attack_indices] = False
    
    batch_allocation_sampler.attack_samples_indices = attack_indices.tolist()
    batch_allocation_sampler.non_attack_samples_indices = non_attack_indices.tolist()
    
    
def get_subset_idx(num_classes=10, samples_per_class_train=500, samples_per_class_test=300):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR10_idx(root=DATASET_PATH, train=True, transform=transform_train, download=True)
    test_dataset = CIFAR10_idx(root=DATASET_PATH, train=False, transform=transform_test, download=True)

    selected_classes = range(num_classes)  # You can change these to the classes you want

    # Create empty lists to store the subset
    train_subset_indices = []
    test_subset_indices = []
    
    # Iterate through the CIFAR-10 dataset to create the subset
    for class_index in selected_classes:
        train_class_indices = [i for i, label in enumerate(train_dataset.targets) if label == class_index]
        test_class_indices = [i for i, label in enumerate(test_dataset.targets) if label == class_index]
        train_class_indices = train_class_indices[:samples_per_class_train]
        test_class_indices = test_class_indices[:samples_per_class_test]
        
        train_subset_indices += train_class_indices
        test_subset_indices += test_class_indices

    
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
    
    
    return train_subset, test_subset


def confidence(logit, target):         # probability margin
    eye = torch.eye(logit.shape[1]).cuda()
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()  # probability of ground truth
    top2_probs = logit.softmax(1).topk(2, largest = True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1 
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach() # probability of 2nd largest
    return  probs_GT - probs_2nd 

def pgd_loss(model,
                x_natural,
                y,
                optimizer,
                step_size,
                epsilon,
                perturb_steps):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_AE = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss_AE, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv.requires_grad_(False)
    # zero gradient
    optimizer.zero_grad()

    loss = F.cross_entropy(model(x_adv), y)

    return loss


def evaluate(args, model, device, eval_set, loader, eval_attack_batches=None):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_total = 0
    model.eval()
    for batch_idx, (data, target, _) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx < eval_attack_batches:
            # run medium-strength gradient attack
            assert args.test_epsilon == 8./255
            assert args.test_num_steps == 20
            assert args.test_step_size == 2./255
            n_correct_adv = pgd(
                model, data, target,
                epsilon=args.test_epsilon,
                num_steps=args.test_num_steps,
                step_size=args.test_step_size,
            )
            adv_correct += n_correct_adv
            adv_total += len(data)
        total += len(data)
    loss /= total
    accuracy = correct / total
    if adv_total == 0:
        robust_accuracy = -1
    else:
        robust_accuracy = adv_correct / adv_total

    logging.info(
        '{}: Clean loss: {:.4f}, '
        'Clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            correct, total, 100.0 * accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))
    return accuracy, robust_accuracy
