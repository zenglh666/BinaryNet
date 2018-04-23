import torch
import resnet
import numpy
resnet18 = resnet.resnet18(pretrained=True)
resnet18_weights_list = []
for param in resnet18.parameters():
    print(type(param.data), param.size())
    resnet18_weights_list.append(param.data.numpy())
numpy.savez('resnet18_weights.npz',tuple(resnet18_weights_list))
