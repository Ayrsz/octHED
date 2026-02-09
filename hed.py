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
from tqdm import tqdm

from datasets import BipedDataset, BsdsDataset
from fourierhed_model import FFCHED

# Customized import.
from hed_model import HED
from octave_conv import OctaveConv
from octhed_model import OCTHED
from exitationhed_model import EXITHED
from utils import (
    AverageMeter,
    Logger,
    load_checkpoint,
    load_pretrained_caffe,
    save_checkpoint,
    save_graph_of_loss,
    write_config_yaml
)
from VGGInitializer import CaffeVGGInitializer, OctaveVGGInitializer, ExitVGGInitializer

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
        train_dataset = BsdsDataset(dataset_dir=args.dataset, split='train', hsv = args.HSV)
        test_dataset = BsdsDataset(dataset_dir=args.dataset, split='test', hsv= args.HSV)
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
    else:
        raise ValueError(f'Invalid model {args.model}')

        
        
    
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
    net_parameters_id = defaultdict(list)

    # Adapted layer settings, fitted to octave layers
    for name, param in net.named_parameters():
        if (
            'conv1' in name
            or 'conv2' in name
            or 'conv3' in name
            or 'conv4' in name
        ) and ('weight' in name):
            print('{:26} lr:    1 decay:1'.format(name))
            net_parameters_id['conv1-4.weight'].append(param)
        elif (
            'conv1' in name
            or 'conv2' in name
            or 'conv3' in name
            or 'conv4' in name
        ) and ('bias' in name):
            print('{:26} lr:    2 decay:0'.format(name))
            net_parameters_id['conv1-4.bias'].append(param)
        elif 'conv5' in name and 'weight' in name:
            print('{:26} lr:  100 decay:1'.format(name))
            net_parameters_id['conv5.weight'].append(param)
        elif 'conv5' in name and 'bias' in name:
            print('{:26} lr:  200 decay:0'.format(name))
            net_parameters_id['conv5.bias'].append(param)
        elif 'score_dsn' in name and 'weight' in name:
            print('{:26} lr: 0.01 decay:1'.format(name))
            net_parameters_id['score_dsn_1-5.weight'].append(param)
        elif 'score_dsn' in name and 'bias' in name:
            print('{:26} lr: 0.02 decay:0'.format(name))
            net_parameters_id['score_dsn_1-5.bias'].append(param)
        elif 'score_final' in name and 'weight' in name:
            print('{:26} lr:0.001 decay:1'.format(name))
            net_parameters_id['score_final.weight'].append(param)
        elif 'score_final' in name and 'bias' in name:
            print('{:26} lr:0.002 decay:0'.format(name))
            net_parameters_id['score_final.bias'].append(param)
        #Residual net
        elif 'octdense' in name and 'weight' in name:
            print('{:26} lr:0.001 decay:0'.format(name))
            net_parameters_id['residual.weight'].append(param)
        elif 'octdense' in name and 'bias' in name:
            print('{:26} lr:0.002 decay:0'.format(name))
            net_parameters_id['residual.bias'].append(param)
        else:
            print('EITA NAO PEGOU', name)
    # Create optimizer.
    opt = torch.optim.SGD(
        [
            {
                'params': net_parameters_id['conv1-4.weight'],
                'lr': args.lr * 1,
                'weight_decay': args.weight_decay,
            },
            {
                'params': net_parameters_id['conv1-4.bias'],
                'lr': args.lr * 2,
                'weight_decay': 0.0,
            },
            {
                'params': net_parameters_id['conv5.weight'],
                'lr': args.lr * 100, #ALTERADO ORIGINAL 100
                'weight_decay': args.weight_decay,
            },
            {
                'params': net_parameters_id['conv5.bias'],
                'lr': args.lr * 200, #ALTERADO ORIGINAL 200
                'weight_decay': 0.0,
            },
            {
                'params': net_parameters_id['score_dsn_1-5.weight'],
                'lr': args.lr * 0.01,
                'weight_decay': args.weight_decay,
            },
            {
                'params': net_parameters_id['score_dsn_1-5.bias'],
                'lr': args.lr * 0.02,
                'weight_decay': 0.0,
            },
            {
                'params': net_parameters_id['score_final.weight'],
                'lr': args.lr * 0.001,
                'weight_decay': args.weight_decay,
            },
            {
                'params': net_parameters_id['score_final.bias'],
                'lr': args.lr * 0.002,
                'weight_decay': 0.0,
            },
            {
                'params': net_parameters_id['residual.weight'],
                'lr': args.lr * 1,
                'weight_decay': args.weight_decay,
            },
            {
                'params': net_parameters_id['residual.bias'],
                'lr': args.lr * 2,
                'weight_decay': 0.0,
            }
        ],
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

    
    if args.fine_tuning and isinstance(net.module, OCTHED): #If is octave
        initializer = OctaveVGGInitializer()
        print("FINE-TUNING OCTHED")
    elif args.fine_tuning and args.vgg16_caffe != "" and isinstance(net.module, HED): #If is the normal model with caffe
        initializer = CaffeVGGInitializer(path=args.vgg16_caffe, only_vgg=True)
        print("FINE-TUNING HED CAFFE")
    elif args.fine_tuning and isinstance(net.module, HED): #If is the normal model without caffe
        initializer = OctaveVGGInitializer()
        print("FINE-TUNING HED")
    elif args.fine_tuning and isinstance(net.module, FFCHED): # If is FFCHED
        raise NotImplementedError('Not implemented yet!')
    elif args.fine_tuning and isinstance(net.module, EXITHED):
        initializer = ExitVGGInitializer()
        print("oi")
    else:
        print('\tWITHOUT FINE TUNING!\t\n')
    
    

    initializer.load(net, device) if args.fine_tuning else 0
    # Resume the checkpoint.
    if args.checkpoint:
        load_checkpoint(net, opt, args.checkpoint)  # Omit the returned values.

    # Resume the HED Caffe model.
    if args.caffe_model:
        load_pretrained_caffe(net, args.caffe_model)


    ################################################
    # V. Training / testing.
    ################################################
    if args.test is True:
        # Only test.
        test(test_loader, net, save_dir=join(output_dir, 'test'))
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
            train_epoch_loss = train(
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
            save_graph_of_loss(train_epoch_losses)

        if args.save_parameters:
            parameters = {}
            parameters['LR'] = args.lr
            parameters['EPOCHS'] = args.max_epoch
            parameters['DATASET'] = str(train_dataset)
            parameters['FINE_TUNING'] = args.fine_tuning    
            parameters['MODEL'] = str(net)
            write_config_yaml(config_dict=parameters)

            


def train(train_loader, net, opt, lr_schd, epoch, save_dir):
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
        batch_loss = sum([
            weighted_cross_entropy_loss(preds, edges) for preds in preds_list
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


def test(test_loader, net, save_dir):
    """Test procedure."""
    # Create the directories.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    save_png_dir = join(save_dir, 'png')
    if not isdir(save_png_dir):
        os.makedirs(save_png_dir)
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
        name = test_loader.dataset.images_name[batch_index]
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
    losses = torch.nn.functional.binary_cross_entropy(
        preds.float(), edges.float(), weight=weight, reduction='none'
    )
    loss = torch.sum(losses) / b
    return loss


if __name__ == '__main__':
    main()
