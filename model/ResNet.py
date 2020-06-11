
import torch
import torch.nn as nn
import sparseconvnet as scn

class ResNet(nn.Module):
    '''
    dimention
    '''
    def __init__(self, dimension = 3, device = 'cuda'):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential(
            scn.SubmanifoldConvolution(dimension, nIn=1, nOut=8, filter_size=3, bias=False),
            scn.MaxPooling(dimension, pool_size=3, pool_stride=2),
            scn.SparseResNet(dimension, 8, [
                        ['b', 8, 2, 1],
                        ['b', 16, 2, 2],
                        ['b', 24, 2, 2],
                        ['b', 32, 2, 2]]),
            scn.Convolution(dimension,  nIn=32, nOut=64, filter_size=5, filter_stride=1, bias=False),
            scn.BatchNormReLU(64),
            scn.SparseToDense(dimension, 64)).to(device)
        self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1]*dimension))
        self.inputLayer = scn.InputLayer(dimension,self.spatial_size,mode=4)
        self.linear = nn.Linear(64, 1).to(device)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, 64)
        x = self.linear(x)
        return x

class Hello(nn.Module):
    def __init__(self, dimension = 3, device = 'cuda'):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.SparseVggNet(dimension, 1,
                             [['C', 8], ['C', 8], ['MP', 3, 2], 
                              ['C', 16], ['C', 16], ['MP', 3, 2], 
                              ['C', 24], ['C', 24], ['MP', 3, 2]])
        ).add(
            scn.SubmanifoldConvolution(dimension, 24, 32, 3, False)
        ).add(
            scn.BatchNormReLU(32)
        ).add(
            scn.SparseToDense(dimension, 32) 
        ).to(device)
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor([1]*dimension))
        self.inputLayer = scn.InputLayer(dimension, self.spatial_size)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        return x