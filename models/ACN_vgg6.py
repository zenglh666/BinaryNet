from nnUtils import *

model = Sequential([
    AccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='conv1', bias=False),
    BatchNormalization(name='batchnorm1'),
    HardTanhReLU(),
    AccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='conv2', bias=False),
    BatchNormalization(name='batchnorm2'),
    HardTanhReLU(),
    AccurateBinarizedWeightOnlySpatialConvolution(128,3,3,2,2, padding='SAME', name='conv3', bias=False),
    BatchNormalization(name='batchnorm3'),
    HardTanhReLU(),
    AccurateBinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='conv4', bias=False),
    BatchNormalization(name='batchnorm4'),
    HardTanhReLU(),
    AccurateBinarizedWeightOnlySpatialConvolution(256,3,3,2,2, padding='SAME', name='conv5', bias=False),
    BatchNormalization(name='batchnorm5'),
    HardTanhReLU(),
    AccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='conv6', bias=False),
    BatchNormalization(name='batchnorm6'),
    HardTanhReLU(),
    SpatialAveragePooling(6,6,1,1, name='pool7'),
    BinarizedWeightOnlyAffine(10, name='fc6', bias=False)
])