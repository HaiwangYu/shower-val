import torch
from torch import nn
import torch.optim as optim

def DistanceLoss(a, b) :
    return torch.dist(a,b)

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9, weight_decay=0)
model.weight.data.fill_(10)
model.bias.data.fill_(0)

x = torch.tensor([3.14])
target = torch.tensor([3.14])
output = model(x)
print('output: ', output)
loss = DistanceLoss(output, target)
loss.backward()
print('grad: ', model.weight.grad)
print('before step', model.weight)
optimizer.step()
print('after step', model.weight)
