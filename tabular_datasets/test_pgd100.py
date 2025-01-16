
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
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--seed', type=int, default=1)

    args = vars(parser.parse_args())
    
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
        'weight_decay': 1e-5
    }

    args['AT'] = {
        'norm': 'l2',
        'epsilon': args['epsilon'],
        'step_size':  args['epsilon']/5,
        'perturb_steps': 10,
    }
    
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

    
    if torch.cuda.device_count() > 1: 
        print('Using nn.DataParallel')
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


    acc_ref = torch.zeros(1).cuda()
    err_ref = torch.ones(1).cuda() * 0.9
    F_Bs = []
    ns = []
    epoch_times = []
    
    checkpoint_path = args['checkpoint_path']
    print('--------------Test Start!--------------')
    
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    val_nat, val_pgd20 = lib.evaluate_pgd_100(model, X_num, Y_device, [lib.TEST], D, 2048, device, step_size=args['AT']['step_size'], epsilon=args['AT']['epsilon'])

    print('Done!')
    
    print(args)