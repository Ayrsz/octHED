import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from os.path import abspath, dirname, isdir, join

import numpy as np
import scipy.io as sio
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from datasets import BipedDataset, BsdsDataset
from dataset_bsds500_uncert import BSDS_UncertLoader

# Customized import.
from models.fourier_model import FFCHED
from models.hed_model import HED
from models.octave_conv import OctaveConv
from models.octave_model import OCTHED
from models.side_outputs_dense_model import DENSEHED
from models.exitationhed_model import EXITHED
from models.decoder_model import UncertHED
from utils import (
    AverageMeter,
    Logger,
    load_checkpoint,
    load_pretrained_caffe,
    save_checkpoint,
    save_graph_of_loss,
    write_config_yaml,
    count_flops,
    return_layer_coficients
)
from VGGInitializer import *
# Parse arguments.
parser = argparse.ArgumentParser(description='HED training.')
# 1. Actions.
parser.add_argument(
    '--test', default=False, help='Only test the model.', action='store_true'
)
parser.add_argument(
    '--graph', default = False, help='Save plot gaph', action = 'store_true',
)
parser.add_argument(
    '--save_parameters', default = False, help = 'Save parametres from training', action = 'store_true'
)
# 2. Counts.
parser.add_argument(
    '--train_batch_size',
    default=1,
    type=int,
    metavar='N',
    help='Training batch size.',
)
parser.add_argument(
    '--test_batch_size',
    default=1,
    type=int,
    metavar='N',
    help='Test batch size.',
)
parser.add_argument(
    '--train_iter_size',
    default=10,
    type=int,
    metavar='N',
    help='Training iteration size.',
)
parser.add_argument(
    '--max_epoch', default=40, type=int, metavar='N', help='Total epochs.'
)
parser.add_argument(
    '--print_freq', default=500, type=int, metavar='N', help='Print frequency.'
)
# 3. Optimizer settings.
parser.add_argument(
    '--lr',
    default=1e-6,
    type=float,
    metavar='F',
    help='Initial learning rate.',
)
parser.add_argument(
    '--lr_stepsize',
    default=1e4,
    type=int,
    metavar='N',
    help='Learning rate step size.',
)
# Note: Step size is based on number of iterations, not number of batches.
#   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L498
parser.add_argument(
    '--lr_gamma',
    default=0.1,
    type=float,
    metavar='F',
    help='Learning rate decay (gamma).',
)
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='F', help='Momentum.'
)
parser.add_argument(
    '--weight_decay',
    default=2e-4,
    type=float,
    metavar='F',
    help='Weight decay.',
)
parser.add_argument(
    '--loss',
    default= 'weight_cross_entropy',
    type = str,
    metavar = 'F',
    help ='weight_cross_entropy or ranked_loss'
)
# 4. Files and folders.
parser.add_argument(
    '--fine_tuning', default = False, help = "If true, set the vgg image net init", action = "store_true"
)

parser.add_argument(
    '--vgg16_caffe', default='', help='Resume VGG-16 Caffe parameters.'
)
parser.add_argument('--checkpoint', default='', help='Resume the checkpoint.')
parser.add_argument(
    '--caffe_model', default='', help='Resume HED Caffe model.'
)

parser.add_argument('--output', default='./output', help='Output folder.')
parser.add_argument(
    '--dataset',
    default='./data/HED-BSDS',
    help='HED-BSDS or BIPED dataset folder.',
)
# 5. Others.
parser.add_argument(
    '--cpu', default=False, help='Enable CPU mode.', action='store_true'
)
# 6. Model
parser.add_argument('--model', default='HED', help='HED or OCTHED')
parser.add_argument(
    '--alpha', default=-1, help='Octave alpha, only use with OCTHED'
)
parser.add_argument(
    '--octave_layers',
    default='',
    type=str,
    help='Multiple args, you could do conv2 conv3 conv4 conv5',
    nargs='+',
)
parser.add_argument(
    '--HSV',
    default = False,
    action = 'store_true',
    help = 'transform images from the dataset to hsv'
)

args = parser.parse_args()

# Set device.
device = torch.device('cpu' if args.cpu else 'cuda')


def main():
    ################################################
    # I. Miscellaneous.
    ################################################
    # Create the output directory.
    current_dir = abspath(dirname(__file__))
    output_dir = join(current_dir, args.output)
    if not isdir(output_dir):
        os.makedirs(output_dir)

    # Set logger.
    now_str = datetime.now().strftime('%y%m%d-%H%M%S')
    log = Logger(join(output_dir, 'log-{}.txt'.format(now_str)))
    sys.stdout = log  # Overwrite the standard output.

    ################################################
    # II. Datasets.
    ################################################
    # Datasets and dataloaders.
    if 'BSDS' in args.dataset.upper():
        #train_dataset = BsdsDataset(dataset_dir=args.dataset, split='train', hsv = args.HSV)
        #test_dataset = BsdsDataset(dataset_dir=args.dataset, split='test', hsv= args.HSV)
        train_dataset = BSDS_UncertLoader(split = 'train')
        test_dataset = BSDS_UncertLoader(split = 'test')
    elif 'BIPED' in args.dataset.upper():
        train_dataset = BipedDataset(dataset_dir=args.dataset, split='train', hsv= args.HSV)
        test_dataset = BipedDataset(dataset_dir=args.dataset, split='test', hsv= args.HSV)
    else:
        raise ValueError('Invalid dataset')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False,
    )

    ################################################
    # III. Network and optimizer.
    ################################################
    # Create the network in GPU.
    if args.model == 'HED':
        net = nn.DataParallel(HED(device))
        net.to(device)
    elif args.model == 'OCTHED':
        net = nn.DataParallel(
            OCTHED(
                device,
                alpha=float(args.alpha),
                octave_layers=args.octave_layers,
            )
        )
        net.to(device)
    elif args.model == 'FFCHED':
        net = nn.DataParallel(
            FFCHED(
                device,
                ratio=float(args.alpha),
                fourier_layer=args.octave_layers,
            )
        )
        net.to(device)
        raise NotImplementedError('Not implemented yet!')
    elif args.model == 'EXITHED':
        net = nn.DataParallel(
            EXITHED(device)
        )
        net.to(device)
    elif args.model == 'DENSEHED':
        net = nn.DataParallel(
            DENSEHED(device)
        )
    elif args.model == 'UNCERTHED':
        net = nn.DataParallel(UncertHED(device))
        net.to(device)
    else:
        raise ValueError(f'Invalid model {args.model}')

    count_flops(net.module)
        
    
    # Initialize the weights for HED model.
    def weights_init(m):
        """Weight initialization function."""
        if isinstance(m, nn.Conv2d):
            # Initialize: m.weight.
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                # Constant initialization for fusion layer in HED network.
                torch.nn.init.constant_(m.weight, 0.2)
            else:
                # Zero initialization following official repository.
                # Reference: hed/docs/tutorial/layers.md
                m.weight.data.zero_()
            # Initialize: m.bias.
            if m.bias is not None:
                # Zero initialization.
                m.bias.data.zero_()
        elif isinstance(m, OctaveConv):
            if m.conv_h2h is not None:
                nn.init.zeros_(m.conv_h2h.weight)
                nn.init.zeros_(m.conv_h2h.bias)

            if m.conv_l2l is not None:
                nn.init.zeros_(m.conv_l2l.weight)
                nn.init.zeros_(m.conv_l2l.bias)

            if m.conv_h2l is not None:
                nn.init.zeros_(m.conv_h2l.weight)
                nn.init.zeros_(m.conv_h2l.bias)

            if m.conv_l2h is not None:
                nn.init.zeros_(m.conv_l2h.weight)
                nn.init.zeros_(m.conv_l2h.bias)

    net.apply(weights_init)
    # Optimizer settings.
    
    
    LAYER_SETTINGS = {
    'conv1':         {'lr': 1,     'wd': 1},
    'conv2':         {'lr': 1,     'wd': 1},
    'conv3':         {'lr': 1,     'wd': 1},
    'conv4':         {'lr': 1,     'wd': 1},
    'conv5':         {'lr': 100,   'wd': 1},
    'score_dsn':     {'lr': 0.01,  'wd': 1},
    'score_final':   {'lr': 0.001, 'wd': 1},
    'exitation':     {'lr': 1,     'wd': 1},
    'deep_conection':{'lr': 2,     'wd': 0},
    'decoderMEAN':   {'lr': 10,   'wd': 1},
    'decoderSTD':    {'lr': 10,   'wd': 1},
    }
    
    net_parameters = return_layer_coficients(net, LAYER_SETTINGS)
    
    optim_params = [
    {
        'params': params,
        'lr': args.lr * lr_m,
        'weight_decay': args.weight_decay * wd_m
    }
    for (lr_m, wd_m), params in net_parameters.items()
    ]

    
    # Create optimizer.
    opt = torch.optim.SGD(
        params=optim_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # Note: In train_val.prototxt and deploy.prototxt, the learning rates of score_final.weight/bias are different.

    # Learning rate scheduler.
    lr_schd = lr_scheduler.StepLR(
        opt, step_size=args.lr_stepsize, gamma=args.lr_gamma
    )

    ################################################
    # IV. Pre-trained parameters.
    ################################################
    # Load parameters from pre-trained VGG-16 Caffe model.
    

    # 2. Lógica de inicialização simplificada
    if not args.fine_tuning:
        tag = "NULL"
    elif isinstance(net.module, OCTHED):
        initializer = OctaveVGGInitializer()
        tag = "OCTHED"
    elif isinstance(net.module, HED):
        if args.vgg16_caffe:
            initializer = CaffeVGGInitializer(path=args.vgg16_caffe, only_vgg=True)
            tag = "HED CAFFE"
        else:
            initializer = OctaveVGGInitializer()
            tag = "HED"
    elif isinstance(net.module, FFCHED):
        raise NotImplementedError('Not implemented yet!')
    elif isinstance(net.module, (EXITHED, DENSEHED)): # Tupla para múltiplos tipos
        initializer = ExitVGGInitializer()
        tag = "EXITHED" 
    elif isinstance(net.module, UncertHED):
        initializer = UncertHEDInitializar()
        tag = "UNCERTHED"
    print(f"FINE-TUNING {tag}")
    

    initializer.load(net, device) if args.fine_tuning else 0
    # Resume the checkpoint.
    if args.checkpoint:
        load_checkpoint(net, opt, args.checkpoint)  # Omit the returned values.

    # Resume the HED Caffe model.
    if args.caffe_model:
        load_pretrained_caffe(net, args.caffe_model)

    if args.loss == 'weight_cross_entropy':
        loss_fn = weighted_cross_entropy_loss

    print("BE CAREFULL! TRAIN AND DATASET UNCERTLY")
    ################################################
    # V. Training / testing.
    ################################################
    if args.test is True:
        # Only test.
        test_uncertly(test_loader, net, save_dir=join(output_dir, 'test'))
    else:
        # Train.
        train_epoch_losses = []
        for epoch in range(args.max_epoch):
            # Initial test.
            if epoch == 0:
                print('Initial test...')
                test(
                    test_loader, net, save_dir=join(output_dir, 'initial-test')
                )
            # Epoch training and test.
            train_epoch_loss = train_uncertly(
                train_loader,
                net,
                opt,
                lr_schd,
                epoch,
                save_dir=join(output_dir, 'epoch-{}-train'.format(epoch)),
            )
            test(
                test_loader,
                net,
                save_dir=join(output_dir, 'epoch-{}-test'.format(epoch)),
            )
            # Write log.
            log.flush()
            # Save checkpoint.
            save_checkpoint(
                state={
                    'net': net.state_dict(),
                    'opt': opt.state_dict(),
                    'epoch': epoch,
                },
                path=os.path.join(
                    output_dir, 'epoch-{}-checkpoint.pt'.format(epoch)
                ),
            )
            # Collect losses.
            train_epoch_losses.append(train_epoch_loss)

        if args.graph:
            save_graph_of_loss(train_epoch_losses, output_dir + "/graph.jpg")

        if args.save_parameters:
            parameters = {}
            parameters['LR'] = args.lr
            parameters['EPOCHS'] = args.max_epoch
            parameters['DATASET'] = train_dataset.__class__.__name__
            parameters['FINE_TUNING'] = args.fine_tuning    
            parameters['MODEL'] = net.__class__.__name__
            write_config_yaml(config_dict=parameters, target_dir = output_dir)

            


def train(train_loader, net, opt, lr_schd, epoch, save_dir, loss_fn):
    
    """Training procedure."""
    # Create the directory.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    # Switch to train mode and clear the gradient.
    net.train()
    opt.zero_grad()
    # Initialize meter and list.
    batch_loss_meter = AverageMeter()
    # Note: The counter is used here to record number of batches in current training iteration has been processed.
    #       It aims to have large training iteration number even if GPU memory is not enough. However, such trick
    #       can be used because batch normalization is not used in the network architecture.
    counter = 0
    for batch_index, (images, edges) in enumerate(tqdm(train_loader)):
        # Adjust learning rate and modify counter following Caffe's way.
        if counter == 0:
            lr_schd.step()  # Step at the beginning of the iteration.
        counter += 1
        # Get images and edges from current batch.
        images, edges = images.to(device), edges.to(device)
        # Generate predictions.
        preds_list = net(images)
        # Calculate the loss of current batch (sum of all scales and fused).
        # Note: Here we mimic the "iteration" in official repository: iter_size batches will be considered together
        #       to perform one gradient update. To achieve the goal, we calculate the equivalent iteration loss
        #       eqv_iter_loss of current batch and generate the gradient. Then, instead of updating the weights,
        #       we continue to calculate eqv_iter_loss and add the newly generated gradient to current gradient.
        #       After iter_size batches, we will update the weights using the accumulated gradients and then zero
        #       the gradients.
        # Reference:
        #   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L230
        #   https://www.zhihu.com/question/37270367
       
        if isinstance(loss_fn, type) and issubclass(loss_fn, torch.autograd.Function):
            batch_loss = sum([
                    loss_fn.apply(preds, edges) for preds in preds_list
                ])
        else:
            batch_loss = sum([
                    loss_fn(preds, edges) for preds in preds_list
                ])
        
        
        eqv_iter_loss = batch_loss / args.train_iter_size

        # Generate the gradient and accumulate (using equivalent average loss).
        eqv_iter_loss.backward()
        if counter == args.train_iter_size:
            opt.step()
            opt.zero_grad()
            counter = 0  # Reset the counter.
        # Record loss.
        batch_loss_meter.update(batch_loss.item())
        # Log and save intermediate images.
        if batch_index % args.print_freq == args.print_freq - 1:
            # Log.
            print(
                (
                    'Training epoch:{}/{}, batch:{}/{} current iteration:{}, '
                    + 'current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}.'
                ).format(
                    epoch,
                    args.max_epoch,
                    batch_index,
                    len(train_loader),
                    lr_schd.last_epoch,
                    batch_loss_meter.val,
                    batch_loss_meter.avg,
                    lr_schd.get_lr(),
                )
            )
            # Generate intermediate images.
            preds_list_and_edges = preds_list + [edges]
            _, _, h, w = preds_list_and_edges[0].shape
            interm_images = torch.zeros((len(preds_list_and_edges), 1, h, w))
            for i in range(len(preds_list_and_edges)):
                # Only fetch the first image in the batch.
                interm_images[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            # Save the images.
            torchvision.utils.save_image(
                interm_images,
                join(save_dir, 'batch-{}-1st-image.png'.format(batch_index)),
            )
    # Return the epoch average batch_loss.
    return batch_loss_meter.avg

def train_uncertly(train_loader : BSDS_UncertLoader, net, opt, lr_schd, epoch, save_dir):
    """Training procedure."""
    # Create the directory.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    # Switch to train mode and clear the gradient.
    net.train()
    opt.zero_grad()
    # Initialize meter and list.
    batch_loss_meter = AverageMeter()
    # Note: The counter is used here to record number of batches in current training iteration has been processed.
    #       It aims to have large training iteration number even if GPU memory is not enough. However, such trick
    #       can be used because batch normalization is not used in the network architecture.
    counter = 0
    for batch_index, (image, label, label_mean, label_std) in enumerate(tqdm(train_loader)):
        # Adjust learning rate and modify counter following Caffe's way.
        if counter == 0:
            opt.step()
            opt.zero_grad()
            lr_schd.step()  # Step at the beginning of the iteration.
        counter += 1
        # Get images and edges from current batch.
        image, label, label_std = image.cuda(), label.cuda(), label_std.cuda()
        mean, std = net(image)


        outputs_dist = torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=std + 0.001), 1)

        outputs = torch.sigmoid(outputs_dist.rsample())
        ada = (epoch + 1) / args.max_epoch
        bce_loss, mask = cross_entropy_loss_RCF(outputs, label, std, ada)

        std_loss = torch.sum((std - label_std) ** 2 * mask)
       # print(std_loss)
        #breakpoint()
        batch_loss = bce_loss + std_loss
        eqv_iter_loss = batch_loss / args.train_iter_size

        # Generate the gradient and accumulate (using equivalent average loss).
        eqv_iter_loss.backward()
        if counter == args.train_iter_size:
            opt.step()
            opt.zero_grad()
            counter = 0  # Reset the counter.
        # Record loss.
        batch_loss_meter.update(batch_loss.item())
        preds_list = [outputs, mean, std]
        # Log and save intermediate images.
        if batch_index % args.print_freq == args.print_freq - 1:
            # Log.
            print((   'Training epoch:{}/{}, batch:{}/{} current iteration:{}, '
                    + 'current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}.'
                ).format(
                    epoch,
                    args.max_epoch,
                    batch_index,
                    len(train_loader),
                    lr_schd.last_epoch,
                    batch_loss_meter.val,
                    batch_loss_meter.avg,
                    lr_schd.get_lr(),
                )
            )
            # Generate intermediate images.
            preds_list_and_edges = preds_list + [label_mean]
            _, _, h, w = preds_list_and_edges[0].shape
            interm_images = torch.zeros((len(preds_list_and_edges), 1, h, w))
            for i in range(len(preds_list_and_edges)):
                # Only fetch the first image in the batch.
                interm_images[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            # Save the images.
            torchvision.utils.save_image(
                interm_images,
                join(save_dir, 'batch-{}-1st-image.png'.format(batch_index)),
            )
            
            # Print max and min value
            
            print(f"TRAINING ITER EPOCH: {epoch}: IDX: {batch_index}",
                  f"\nMEAN MAX = {torch.max(mean)}, MEAN MIN = {torch.min(mean)}",
                  f"\nSTD MAX = {torch.max(std)}, STD MIN = {torch.min(std)}")

    # Return the epoch average batch_loss.
    return batch_loss_meter.avg

def test(test_loader, net, save_dir):
    """Test procedure."""
    # Create the directories.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    save_png_dir = join(save_dir, 'png')
    if not isdir(save_png_dir):
        os.makedirs(save_png_dir)
    print(save_dir)
    save_mat_dir = join(save_dir, 'mat')
    if not isdir(save_mat_dir):
        os.makedirs(save_mat_dir)
    # Switch to evaluation mode.
    net.eval()
    # Generate predictions and save.
    assert (
        args.test_batch_size == 1
    )  # Currently only support test batch size 1.
    for batch_index, images in enumerate(tqdm(test_loader)):
        images = images.cuda()
        _, _, h, w = images.shape
        preds_list = net(images)
        fuse = preds_list[-1].detach().cpu().numpy()[0, 0]  # Shape: [h, w].
        name : str = test_loader.dataset.filelist[batch_index]
        name = (name.split('/')[-1]).split('.')[0]
        sio.savemat(
            join(save_mat_dir, '{}.mat'.format(name)), {'image_data': fuse}
        )
        Image.fromarray((fuse * 255).astype(np.uint8)).save(
            join(save_png_dir, '{}.png'.format(name))
        )
        # print('Test batch {}/{}.'.format(batch_index + 1, len(test_loader)))

def test_uncertly(test_loader, net, save_dir):
    """Test procedure."""
    # Create the directories.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    save_png_dir = join(save_dir, 'png')
    if not isdir(save_png_dir):
        os.makedirs(save_png_dir)
    print(save_dir)
    save_mat_dir = join(save_dir, 'mat')
    if not isdir(save_mat_dir):
        os.makedirs(save_mat_dir)
    # Switch to evaluation mode.
    net.eval()
    # Generate predictions and save.
    assert (
        args.test_batch_size == 1
    )  # Currently only support test batch size 1.

    for batch_index, image in enumerate(tqdm(test_loader)):

        image, = image.cuda()
        mean, std = net(image)


        outputs_dist = torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=std + 0.001), 1)
        fuse = torch.sigmoid(outputs_dist.rsample()).detach().cpu().numpy().squeeze()

        name : str = test_loader.dataset.filelist[batch_index]
        name = (name.split('/')[-1]).split('.')[0]
        sio.savemat(
            join(save_mat_dir, '{}.mat'.format(name)), {'image_data': fuse}
        )
        Image.fromarray((fuse * 255).astype(np.uint8)).save(
            join(save_png_dir, '{}.png'.format(name))
        )


        # print('Test batch {}/{}.'.format(batch_index + 1, len(test_loader)))
    

def weighted_cross_entropy_loss(preds, edges):
    """Calculate sum of weighted cross entropy loss."""
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos  # Shape: [b,].
    weight = torch.zeros_like(mask)
    weight[edges > 0.5] = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    losses = F.binary_cross_entropy(
        preds.float(), edges.float(), weight=weight, reduction='none'
    )
    loss = torch.sum(losses) / b
    return loss

def cross_entropy_loss_RCF(prediction, labelef, std, ada):
    label = labelef.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()
    num_two = torch.sum((mask == 2).float()).float()
    assert (
        num_negative + num_positive + num_two
        == label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
    )
    assert num_two == 0
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    new_mask = mask * torch.exp(std * ada)
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=new_mask.detach(), reduction='sum'
    )

    return cost, mask

if __name__ == '__main__':
    main()
