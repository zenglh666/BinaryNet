from nnUtils import *

model = Sequential([
    SpatialConvolution(96,11,11,4,4, padding='SAME', name='conv1'),
    ReLU(),
    LocalResposeNormalize(2, 1e-05, 0.75, name='lrn1'),
    SpatialMaxPooling(3,3,2,2, name='pool1'),
    SpatialConvolution(256,5,5, padding='SAME', name='conv2'),
    ReLU(),
    SpatialMaxPooling(3,3,2,2, name='pool2'),
    LocalResposeNormalize(2, 1e-05, 0.75, name='lrn2'),
    SpatialConvolution(384,3,3, padding='SAME', name='conv3'),
    ReLU(),
    SpatialConvolution(384,3,3, padding='SAME', name='conv4'),
    ReLU(),
    SpatialConvolution(256,3,3, padding='SAME', name='conv5'),
    SpatialMaxPooling(3,3,2,2, name='pool5'),
    ReLU(),
    Affine(4096, name='fc1'),
    ReLU(),
    Dropout(0.5),
    Affine(4096, name='fc2'),
    ReLU(),
    Dropout(0.5),
    Affine(1000, name='fc3')
])
