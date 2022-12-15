# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
from models.wrn import WideResNet

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.tinyimages_80mn_loader import TinyImages
    from utils.validation_dataset import validation_split
    from utils.id_dataset import get_misclassified_examples

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--method', type=str, choices=['ours', 'oe'],
                    help='Choose between our method or outlier exposure.')
parser.add_argument('--id_dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--ood_dataset', type=str, choices=['cifar100', 'cifar10', 'svhn'],
                    help='Choose between CIFAR-100, SVHN.')
parser.add_argument('--num_id', type=int, help='Number of ID examples in the test set.')
parser.add_argument('--num_ood', type=int, help='Number of OOD examples in the test set.')
parser.add_argument('--misclassified_id', action='store_true', help='Include misclassified ID examples in the unlabeled set.')
parser.add_argument('--no_test_id', action='store_true', help='No test ID examples in the unlabeled set.')
parser.add_argument('--dual_descent', action='store_true', help='Use dual descent to find optimal weight of confidence term.')
parser.add_argument('--xent_loss_target', type=float, help='Loss target of xent loss.')
parser.add_argument('--model', '-m', type=str, default='allconv',
                    choices=['allconv', 'wrn'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/oe_tune', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/baseline', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


# Use train transform on test_data_in and test_data_ood because we use them in the unlabeled set.
if args.id_dataset == 'cifar10':
    train_data_in = dset.CIFAR10('/iris/u/cchoi1/exploring_ood/data/CIFAR10', train=True, transform=train_transform)
    test_data_in = dset.CIFAR10('/iris/u/cchoi1/exploring_ood/data/CIFAR10', train=False, transform=train_transform)
    num_classes = 10
elif args.id_dataset == 'cifar100':
    train_data_in = dset.CIFAR100('/iris/u/cchoi1/exploring_ood/data/CIFAR100', train=True, transform=train_transform)
    test_data_in = dset.CIFAR100('/iris/u/cchoi1/exploring_ood/data/CIFAR100', train=False, transform=train_transform)
    num_classes = 100
else:
    raise ValueError("ID Dataset must be cifar10 or cifar100")

if args.ood_dataset == 'cifar100':
    train_data_ood = dset.CIFAR100('/iris/u/cchoi1/exploring_ood/data/CIFAR100', train=True, transform=train_transform)
    test_data_ood = dset.CIFAR100('/iris/u/cchoi1/exploring_ood/data/CIFAR100', train=False, transform=train_transform)
elif args.ood_dataset == 'cifar10':
    train_data_ood = dset.CIFAR10('/iris/u/cchoi1/exploring_ood/data/CIFAR10', train=True, transform=train_transform)
    test_data_ood = dset.CIFAR10('/iris/u/cchoi1/exploring_ood/data/CIFAR10', train=False, transform=train_transform)
elif args.ood_dataset == 'svhn':
    train_data_ood = dset.SVHN('/iris/u/cchoi1/exploring_ood/data/SVHN', split='train', transform=train_transform)
    test_data_ood = dset.SVHN('/iris/u/cchoi1/exploring_ood/data/SVHN', split='test', transform=train_transform)
else:
    raise ValueError("ID Dataset must be cifar10 or cifar100")


calib_indicator = ''
if args.calibration:
    train_data_in, val_data_in = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'


# Create model
if args.model == 'allconv':
    net = AllConvNet(num_classes)
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

# Restore model
model_found = False
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.id_dataset + calib_indicator + '_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders


# Identify misclassified ID train examples
misclassified_id_indicator = ''
if args.misclassified_id:
    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=1, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
    misclassified_idxs = get_misclassified_examples(net, train_loader_in)
    misclassified_id_set = torch.utils.data.Subset(train_data_in, misclassified_idxs)
    print(f"Number of Misclassified ID Examples: {len(misclassified_id_set)}")
    misclassified_id_indicator = '_missid'

# Create unlabeled set
no_test_id_indicator = ''
if args.method == 'ours':
    if args.num_id > 0:
        _, test_data_in = validation_split(test_data_in, val_share=float(args.num_id / len(test_data_in)))
    if args.num_ood > 0:
        _, test_data_ood = validation_split(test_data_ood, val_share=float(args.num_ood / len(test_data_ood)))
    unlabeled_set = [test_data_ood, test_data_in]
    if args.no_test_id:
        unlabeled_set = [test_data_ood]
        no_test_id_indicator = '_notestid'
    if args.misclassified_id:
        unlabeled_set.append(misclassified_id_set)
    unlabeled_set = torch.utils.data.ConcatDataset(unlabeled_set)
else:
    unlabeled_set = TinyImages(transform=trn.Compose(
        [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
        trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))

id_ood_ratio = 'id_ood' + str(args.num_id / args.num_ood)

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    unlabeled_set,
    batch_size=args.oe_batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

id_test_loader = torch.utils.data.DataLoader(
    test_data_in,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)


optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

dual_descent_identifier = ''
if args.dual_descent:
    lambda_w = nn.parameter.Parameter(
        torch.ones(1, requires_grad=True, device="cuda")
    )
    lambda_opt = torch.optim.SGD(
        [lambda_w], state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)
    lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
        lambda_opt,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))
    dual_descent_identifier = f"_dual{args.xent_loss_target}"

def get_xent_loss_limit():
    net.eval()
    loss_avg = 0.0
    xent_losses = []
    for data, target in train_loader_in:
        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)
        loss = F.cross_entropy(x, target)
        xent_losses.append(loss.item())

    print('min xent loss', np.min(xent_losses))
    print('med xent loss', np.median(xent_losses))
    print('max xent loss', np.max(xent_losses))
    print('avg xent loss', np.mean(xent_losses))

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    batch_idx = 0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()

        xent_loss = F.cross_entropy(x[:len(in_set[0])], target)
        if args.dual_descent:
            loss = xent_loss + F.softplus(lambda_w).squeeze() * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            if batch_idx % 1000 == 0:
                lambda_scheduler.step()
                lambda_opt.zero_grad()
                if xent_loss.item() < args.xent_loss_target:
                    # increase lambda since xent loss is low enough
                    (-1 * loss).backward()
                else:
                    # decrease lambda since xent loss is too high
                    loss.backward()
                lambda_opt.step()
            else:
                loss.backward()
                optimizer.step()
        else:
            # cross-entropy from softmax distribution to uniform distribution
            loss = xent_loss + 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            loss.backward()
            optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        batch_idx += 1

    state['train_loss'] = loss_avg
    print('avg loss', loss_avg)

# test function
def test(test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    print("Testing on ID dataset...")
    test(id_test_loader)
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.id_dataset + '_' + args.ood_dataset + '_' + id_ood_ratio + misclassified_id_indicator + no_test_id_indicator + dual_descent_identifier + calib_indicator + '_' + args.model +
                                  '_oe_tune_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
get_xent_loss_limit()
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test(id_test_loader)

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.id_dataset + '_' + args.ood_dataset + '_' + id_ood_ratio + misclassified_id_indicator + no_test_id_indicator + dual_descent_identifier + calib_indicator + '_' + args.model +
                            '_oe_tune_epoch_' + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.id_dataset + '_' + args.ood_dataset + '_' + id_ood_ratio + misclassified_id_indicator + no_test_id_indicator + dual_descent_identifier + calib_indicator + '_' + args.model +
                             '_oe_tune_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.id_dataset + '_' + args.ood_dataset + '_' + id_ood_ratio + misclassified_id_indicator + no_test_id_indicator + dual_descent_identifier + calib_indicator + '_' + args.model +
                                      '_oe_tune_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )

if args.dual_descent:
    print('final lambda', F.softplus(lambda_w).item())

if args.calibration:
    id_val_loader = torch.utils.data.DataLoader(
        val_data_in,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
    test(id_val_loader)
    print(state)
