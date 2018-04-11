from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='conv1', bias=False),
    BatchNormalization(name='batchnorm1'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='conv2', bias=False),
    BatchNormalization(name='batchnorm2'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(128,3,3,2,2, padding='SAME', name='conv3', bias=False),
    BatchNormalization(name='batchnorm3'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='conv4', bias=False),
    BatchNormalization(name='batchnorm4'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(256,3,3,2,2, padding='SAME', name='conv5', bias=False),
    BatchNormalization(name='batchnorm5'),
    HardTanh(),
    BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='conv6', bias=False),
    BatchNormalization(name='batchnorm6'),
    HardTanh(),
    SpatialAveragePooling(6,6,1,1, name='pool7'),
    BinarizedWeightOnlyAffine(10, name='fc6')
])