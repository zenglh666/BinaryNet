from nnUtils import *

model = Sequential([
    MoreAccurateBinarizedWeightOnlySpatialConvolution(64,7,7,2,2, padding='SAME', name='conv1'),
    BatchNormalization(name='batch1'),
    ReLU(name='relu1'),
    SpatialMaxPooling(3,3,2,2, name='pool1', padding='SAME'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res2_1'),
    ReLU(name='rrelu2'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res2_2'),
    ReLU(name='rrelu2'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(128,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], connect=False, name='res3_1',
        mapfunc=MoreAccurateBinarizedWeightOnlySpatialConvolution(128,1,1,2,2, padding='SAME', name='rconv3')),
    ReLU(name='rrelu2'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res3_2'),
    ReLU(name='rrelu2'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(256,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], connect=False, name='res4_1',
        mapfunc=MoreAccurateBinarizedWeightOnlySpatialConvolution(256,1,1,2,2, padding='SAME', name='rconv3')),
    ReLU(name='rrelu2'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res4_2'),
    ReLU(name='rrelu2'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(512,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], connect=False, name='res5_1',
        mapfunc=MoreAccurateBinarizedWeightOnlySpatialConvolution(512,1,1,2,2, padding='SAME', name='rconv3')),
    ReLU(name='rrelu2'),
    ResidualV3([
        MoreAccurateBinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        MoreAccurateBinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res5_2'),
    ReLU(name='rrelu2'),
    SpatialAveragePooling(7,7,1,1, name='pool5'),
    Affine(1000, name='fc6')
])