from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='conv1'),
    BatchNormalization(name='batchnorm1'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='conv2'),
    BatchNormalization(name='batchnorm2'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(128,3,3,2,2, padding='SAME', name='conv3'),
    BatchNormalization(name='batchnorm3'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='conv4'),
    BatchNormalization(name='batchnorm4'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(256,3,3,2,2, padding='SAME', name='conv5'),
    BatchNormalization(name='batchnorm5'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='conv6'),
    BatchNormalization(name='batchnorm6'),
    HardTanh(),
    SpatialAveragePooling(6,6,1,1, name='pool7'),
    Affine(10, name='fc6')
])