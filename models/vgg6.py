from nnUtils import *

model = Sequential([
    SpatialConvolution(64,3,3, padding='SAME', name='conv1'),
    ReLU(name='rrelu1'),
    SpatialConvolution(64,3,3, padding='SAME', name='conv2'),
    ReLU(name='rrelu2'),
    SpatialConvolution(128,3,3,2,2, padding='SAME', name='conv3'),
    ReLU(name='rrelu3'),
    SpatialConvolution(128,3,3, padding='SAME', name='conv4'),
    ReLU(name='rrelu4'),
    SpatialConvolution(256,3,3,2,2, padding='SAME', name='conv5'),
    ReLU(name='rrelu5'),
    SpatialConvolution(256,3,3, padding='SAME', name='conv6'),
    ReLU(name='rrelu6'),
    SpatialAveragePooling(6,6,1,1, name='pool5'),
    Affine(10, name='fc6')
])