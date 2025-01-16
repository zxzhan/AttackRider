
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import zero
from torch import Tensor
import argparse

from ft_transformer import Transformer
import os
import lib

from torch.backends import cudnn
import random
import time


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='jannis')
    parser.add_argument('--e', type=int, default=-1)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)

    args = vars(parser.parse_args())  # Convert Namespace to dictionary
    # args['seed'] = 1
    
    args['data'] = {
        'normalization': 'minmax',
        'path': './data/' + args['dataset'],
    }
    
    args['model'] = {
        'activation': 'reglu',
        'attention_dropout': 0.2,
        'd_ffn_factor': 1.333333333333333,
        'd_token': 192,
        'ffn_dropout': 0.1,
        'initialization': 'kaiming',
        'n_heads': 8,
        'n_layers': 3,
        'prenormalization': True,
        'residual_dropout': 0.0
    }
    args['model'].setdefault('token_bias', True)
    args['model'].setdefault('kv_compression', None)
    args['model'].setdefault('kv_compression_sharing', None)
    
    args['training'] = {
        'batch_size': 1024 if args['dataset'] == 'covtype' else 512,
        'eval_batch_size': 2048,
        'lr': 0.0001,
        'lr_n_decays': 0,
        'n_epochs': 100,
        'optimizer': 'adamw',
        'patience': 9999,
        'weight_decay': 1e-5 #5e-04 # 
    }

    args['AT'] = {     # L2 norm, epsilon = 0.1, Now Fixed
        'norm': 'l2',
        'epsilon': args['epsilon'], # 0.1,
        'step_size':  args['epsilon']/5, # 0.02, # 
        'perturb_steps': 10,
    }

    args['Bullet'] = {
        'perturb_steps_prime': 2,
        'step_size_prime': args['AT']['step_size']*3, # 0.06,
        'gamma': 0.5,
    }
    
    args['AtkRider'] = {
        'm_prime': args['e'] * args['training']['batch_size'],
    }
    
    
    args['output'] = './dbac_output/' + args['dataset'] + '/' + args['algorithm'] + '_seed' + str(args['seed'])
    
    args['output'] = './dbac_output/' + args['dataset'] + '/' + args['algorithm']  + '_e' + str(args['e']) + '_ep' + str(args['AT']['epsilon']) + '_seed' + str(args['seed'])
    if args['gamma'] != 0.5:
        args['output'] = './dbac_output/' + args['dataset'] + '/' + args['algorithm']  + '_e' + str(args['e']) + '_ep' + str(args['AT']['epsilon']) + '_gamma' + str(args['Bullet']['gamma']) + '_seed' + str(args['seed'])
    os.makedirs(args['output'], exist_ok=True)
    return args


if __name__ == "__main__":
    args = load_args()

    SEED = args['seed']
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    D = lib.Dataset.from_dir(args['data']['path'])
    
    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    
    
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = torch.device('cuda')
    
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat = X
    del X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = args['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']
    chunk_size = None

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    
    model = Transformer(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories(X_cat),
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **args['model'],
    ).to(device)

    
    model = nn.DataParallel(model)
    args['n_parameters'] = lib.get_n_parameters(model)

    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    checkpoint_path = args['output'] + '/checkpoint.pt'

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'epoch_size': epoch_size,
                    'n_parameters': args['n_parameters'],
                }.items()
            )
        )

    def apply_model(part, idx):
        return model(
            None if X_num is None else X_num[part][idx],
            None if X_cat is None else X_cat[part][idx],
        )

    @torch.no_grad()
    def evaluate(parts):
        global eval_batch_size
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in lib.IndexLoader(
                                    D.size(part), eval_batch_size, False, device
                                )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    print('New eval batch size:', eval_batch_size)
                    args['eval_batch_size'] = eval_batch_size
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                'logits',
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
        return metrics, predictions

    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            checkpoint_path,
        )
        lib.backup_output(args['output'])

    print(args)
    
    def get_FB_epoch(epochs): 
        all_start_epoch = epochs * 0.9   
        F_B = [0.5 / all_start_epoch * epoch + 0.5 if epoch<=all_start_epoch else 1 for epoch in range(epochs)] 
        F_B = [-1] + F_B
        return F_B

    F_B = get_FB_epoch(args['training']['n_epochs'])
    print(F_B)
    
    acc_ref = torch.zeros(1).cuda()
    err_ref = torch.ones(1).cuda() * 0.9
    F_Bs = []
    ns = []
    epoch_times = []
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()
        F_B_epoch = F_B[stream.epoch]
        p_percent = F_B_epoch      
        if args['AtkRider']['m_prime']<0:
            n = -args['AtkRider']['m_prime']

        
        boundary_size = math.ceil(F_B_epoch*n*args['training']['batch_size'])
        print('Boundary_size:', boundary_size)
        if stream.epoch!=1: 
            print("F_B = {:.4f} | n = {:d} | Boundary_size = {:d}".format(F_B_epoch, n, boundary_size))
            print('F_B mean: {:.4f}'.format(np.mean(F_Bs)))
            print("n = {:d}".format(n))
            print('n mean: {:.4f}'.format(np.mean(ns)))
        
        batch_count = 0
        
        model_update_times = 0
        
        start_time = time.time()
        model.train()
        epoch_losses = []
        for ii, batch_idx in enumerate(epoch):
            batch_count += 1
            
            p_percent = F_B_epoch
            if n==1:
                current_batch_idx = batch_idx
            elif batch_count % n == 1: # the first batch
                current_batch_idx = batch_idx
                continue
            else:  # the following batches
                current_batch_idx = torch.cat((current_batch_idx, batch_idx), 0)
                
                if batch_count % n != 0 and ii != epoch_size-1:
                    continue
            
            batch_count = 0
            ns.append(n)
            
            loss = lib.train_at_dbac_atkrider_batch(
                optimizer,
                loss_fn,
                model=model,
                x_natural=X_num[lib.TRAIN][current_batch_idx], 
                y=Y_device[lib.TRAIN][current_batch_idx],  
                step_size=args['AT']['step_size'],
                epsilon=args['AT']['epsilon'],
                perturb_steps=args['AT']['perturb_steps'],
                norm = args['AT']['norm'],
                train_batch_size=args['training']['batch_size'],
                model_update_times=model_update_times,
                boundary_size = boundary_size,
            )
            F_Bs.append(F_B_epoch)
            epoch_losses.append(loss.detach())
        
        epoch_time = time.time() - start_time
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 4)}')
        print(f'Epoch time: {epoch_time:.2f}s')
        epoch_times.append(epoch_time)
        val_nat, val_pgd20 = lib.evaluate_pgd(model, X_num, Y_device, [lib.TRAIN, lib.VAL, lib.TEST], D, 2048, device, step_size=args['AT']['step_size'], epsilon=args['AT']['epsilon'], eval_num=4096)
        

        progress.update(val_pgd20)
        
        if progress.success:
            print('New best epoch!')
            args['best_epoch'] = stream.epoch
            if 'Standard' in args['algorithm']:
                args['metrics'] = val_nat
            elif 'AT' in args['algorithm'] or 'TRADES' in args['algorithm']:
                args['metrics'] = val_pgd20
            save_checkpoint(False)

        elif progress.fail:
            break
        
    
    print('--------------Training finished!--------------')
    print("Total Time: {:.4f} mins".format(np.sum(epoch_times)/60))
    
    print('\nRunning the final evaluation...')
    print('Best epoch:', args['best_epoch'])
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    val_nat, val_pgd20 = lib.evaluate_pgd_100(model, X_num, Y_device, [lib.TRAIN, lib.VAL, lib.TEST], D, 2048, device, step_size=args['AT']['step_size'], epsilon=args['AT']['epsilon'])

    print('F_Bs mean: ', np.mean(F_Bs))
    print('ns mean: ', np.mean(ns))
    save_checkpoint(True)
    print('Done!')
    
    print(args)