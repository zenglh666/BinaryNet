from nnUtils import *

model = Sequential([
    SpatialConvolution(96,11,11,4,4, padding='VALID', name='conv1'),
    ReLU(),
    SpatialMaxPooling(3,3,2,2, padding='VALID', name='pool1'),
    LocalResposeNormalize(5, 1e-05, 0.75, name='lrn1'),
    SpatialConvolution(256,5,5, padding='SAME', name='conv2'),
    ReLU(),
    SpatialMaxPooling(3,3,2,2, padding='VALID', name='pool2'),
    LocalResposeNormalize(5, 1e-05, 0.75, name='lrn2'),
    SpatialConvolution(384,3,3, padding='SAME', name='conv3'),
    ReLU(),
    SpatialConvolution(384,3,3, padding='SAME', name='conv4'),
    ReLU(),
    SpatialConvolution(256,3,3, padding='SAME', name='conv5'),
    ReLU(),
    SpatialMaxPooling(3,3,2,2, padding='VALID', name='pool5'),
    Affine(4096, name='fc1'),
    ReLU(),
    Dropout(0.5),
    Affine(4096, name='fc2'),
    ReLU(),
    Dropout(0.5),
    Affine(1000, name='fc3')
])
