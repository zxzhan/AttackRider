import torch
import torchvision

from models.resnet import ResNet18
from models.wideresnet import WideResNet

import argparse
DATASET_PATH = '../../CIFAR-10_AT/cifar-data'


parser = argparse.ArgumentParser()
parser.add_argument('--model-path', default=None, type=str, )
parser.add_argument('--log', default=None, type=str, )
parser.add_argument('--eps', default=8/255, type=float, )
parser.add_argument('--dataset', default="CIFAR-10", type=str, )
parser.add_argument('--arch', default="RN", type=str, )

args = parser.parse_args()

args.log = args.model_path.replace(".pt", "_aa.txt".format(args.eps))
print("epsilon:", args.eps)

if args.dataset == "CIFAR-10":
    testset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, transform=torchvision.transforms.ToTensor())
elif args.dataset == "CIFAR-100":
    testset = torchvision.datasets.CIFAR100(DATASET_PATH, train=False, transform=torchvision.transforms.ToTensor())
# elif args.dataset == "svhn":
#     testset = torchvision.datasets.SVHN(DATASET_PATH, split="test", download=False, transform=torchvision.transforms.ToTensor())
elif args.dataset == "TinyImageNet":
    from TinyImageNet_utils.TinyImageNet import TinyImageNet
    from TinyImageNet_utils.resnet_tinyimagenet import ResNet18_tinyimagenet
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
        torchvision.transforms.ToTensor(),
    ])
    testset = TinyImageNet('.', 'val', transform=transform_test, in_memory=True)
    
args.test_batch_size = 2000

testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=len(testset),
                                            num_workers=8, pin_memory=True)

num_classes = 10 if args.dataset != "CIFAR-100" else 100
if args.arch == "RN":
    model = ResNet18(num_classes=num_classes) if args.dataset != "TinyImageNet" else ResNet18_tinyimagenet()
    model.cuda()

model = torch.nn.DataParallel(model)

state_dict = torch.load(args.model_path)["state_dict"]
model.load_state_dict(state_dict)
model.eval()
    
from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=args.eps, log_path=args.log, version='standard')
for i, (x_test, y_test) in enumerate(testloader):
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.test_batch_size)