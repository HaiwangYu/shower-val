#!/usr/bin/env python

import argparse
import os
import numpy as np

import torch
import torchvision

from model import Hello
from model import ResNet
from model import DeepVtx

from timeit import default_timer as timer
import csv
import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                              " (default : 'MODEL.pth')")
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use cuda version of the net",
                        default=False)

    return parser.parse_args()
def count_params(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params = ', params)

if __name__ == "__main__":
    args = get_args()
    input_channels = 3
    output_channels = 1
    
    device = 'cpu'

    nIn = 1
    net = DeepVtx(dimension=3, nIn=nIn, device=device)

    # count_params(net)

    npoints = 10
    coord = torch.randint(0, 1024, [npoints,3])
    ft = torch.randn(npoints, nIn)
    example = [coord, ft]
    
    net.load_state_dict(torch.load(args.model, map_location='cpu'))
    sm = torch.jit.trace(net, example)
    output = net(example)
    print(output[0][0][0])

    sm.save('ts-model.ts')