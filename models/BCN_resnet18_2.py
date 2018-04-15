from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(64,7,7,2,2, padding='SAME', name='conv1', bias=False),
    BatchNormalization(name='batch1'),
    ReLU(name='ReLU1'),
    SpatialMaxPooling(3,3,2,2, name='pool1', padding='SAME'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], name='res2_1'),
    ReLU(name='rReLU2'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], name='res2_2'),
    ReLU(name='rReLU2'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(128,3,3,2,2, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], connect=False, name='res3_1'),
    ReLU(name='rReLU2'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], name='res3_2'),
    ReLU(name='rReLU2'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(256,3,3,2,2, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], connect=False, name='res4_1'),
    ReLU(name='rReLU2'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], name='res4_2'),
    ReLU(name='rReLU2'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(512,3,3,2,2, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], connect=False, name='res5_1'),
    ReLU(name='rReLU2'),
    Residual([
        BinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv1', bias=False),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rReLU1'),
        BinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv2', bias=False),
        BatchNormalization(name='rbatch2'),], name='res5_2'),
    ReLU(name='rReLU2'),
    SpatialAveragePooling(7,7,1,1, name='pool5'),
    BinarizedWeightOnlyAffine(1000, name='fc6')
])