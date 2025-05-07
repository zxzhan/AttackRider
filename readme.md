# AttackRider
Code for our IJCAI-25 paper ```Accelerating Adversarial Training on Under-Utilized GPU```

## Image Datasets

### Requirements
Python 3.10, torch==1.13, numpy==1.26, torchvision==0.14

### Datasets
Download CIFAR-10, CIFAR-100 and TinyImageNet, and change the ```dataset_path``` in the code accordingly. 

### Running Examples
#### Training
```
cd image_datasets/
python train_attackrider_bullet.py --dataset CIFAR-10 --base-at Bullet_TRADES --e 2
python train_attackrider_bullet.py --dataset CIFAR-10 --base-at Bullet_PGDAT --e 2
python train_attackrider_dbac.py --dataset CIFAR-10 --e 6
python train_attackrider_baseat.py --dataset CIFAR-10 --base-at TRADES --e 6
python train_attackrider_baseat.py --dataset CIFAR-10 --base-at PGDAT --e 6
```
where ```--dataset``` can also be CIFAR-100 or TinyImageNet.

#### Evaluation with AutoAttack
```
python test_aa.py --model-path ./CIFAR-10_results/AR_Bullet_TRADES_beta10.0_e2/ep_best.pt --dataset CIFAR-10
```

## Tabular Datasets

### Requirements
Python 3.8, torch==1.13, numpy==1.19.2, torchvision==0.14

Setup PyTorch environment in the same way as https://github.com/yandex-research/rtdl-revisiting-models 

### Datasets
Download Jannis and CoverType from https://github.com/yandex-research/rtdl-revisiting-models and put them under ```./tabular_datasets/data/```

### Running Examples
#### Training
```
cd tabular_datasets/
python train_attackrider_bullet.py --dataset jannis --epsilon 0.1 --e 2 
python train_attackrider_dbac.py --dataset jannis --epsilon 0.1 --e 6
python train_attackrider_baseat.py --dataset jannis --epsilon 0.1 --e 6
python train_attackrider_bullet.py --dataset covtype --epsilon 0.05 --e 2
python train_attackrider_dbac.py --dataset covtype --epsilon 0.05 --e 3
python train_attackrider_baseat.py --dataset covtype --epsilon 0.05 --e 3
```

#### Evaluation with PGD-100 Attack
```
python test_pgd100.py --checkpoint-path ./output/covtype/FT-Transformer_AT_e3/checkpoint.pt --dataset covtype --epsilon 0.05
```

## Reference
Part of the code is based on the following repo:
- https://github.com/yaodongyu/TRADES
- https://github.com/qizhangli/ST-AT
- https://github.com/yandex-research/rtdl-revisiting-models
- https://openreview.net/forum?id=eAPrmf2g8f2
