from nnUtils import *

model = Sequential([
    MoreAccurateBinarizedWeightOnlySpatialConvolution(96,11,11,4,4, padding='VALID', name='conv1', bias=True),
    ReLU(),
    SpatialMaxPooling(3,3,2,2, padding='VALID', name='pool1'),
    LocalResposeNormalize(5, 1e-04, 0.75, name='lrn1'),
    MoreAccurateBinarizedWeightOnlySpatialConvolution(256,5,5, padding='SAME', name='conv2', bias=True),
    ReLU(),
    SpatialMaxPooling(3,3,2,2, padding='VALID', name='pool2'),
    LocalResposeNormalize(5, 1e-04, 0.75, name='lrn2'),
    MoreAccurateBinarizedWeightOnlySpatialConvolution(384,3,3, padding='SAME', name='conv3', bias=True),
    ReLU(),
    MoreAccurateBinarizedWeightOnlySpatialConvolution(384,3,3, padding='SAME', name='conv4', bias=True),
    ReLU(),
    MoreAccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='conv5', bias=True),
    ReLU(),
    SpatialMaxPooling(3,3,2,2, padding='VALID', name='pool5'),
    Affine(4096, name='fc1', bias=True),
    ReLU(),
    Dropout(0.5),
    Affine(4096, name='fc2', bias=True),
    ReLU(),
    Dropout(0.5),
    Affine(1000, name='fc3', bias=True)
])
