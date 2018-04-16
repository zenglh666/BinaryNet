from nnUtils import *

model = Sequential([
    SpatialConvolution(64,7,7,2,2, padding='SAME', name='conv1'),
    BatchNormalization(name='batch1'),
    ReLU(name='relu1'),
    SpatialMaxPooling(3,3,2,2, name='pool1', padding='SAME'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(64,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(64,3,3, padding='SAME', name='rconv2'),], name='res2_1'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(64,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(64,3,3, padding='SAME', name='rconv2'),], name='res2_2'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(128,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(128,3,3, padding='SAME', name='rconv2'),], connect=False, name='res3_1'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(128,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(128,3,3, padding='SAME', name='rconv2'),], name='res3_2'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(256,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(256,3,3, padding='SAME', name='rconv2'),], connect=False, name='res4_1'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(256,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(256,3,3, padding='SAME', name='rconv2'),], name='res4_2'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(512,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(512,3,3, padding='SAME', name='rconv2'),], connect=False, name='res5_1'),
    Residual([
        BatchNormalization(name='rbatch1'),
        ReLU(name='rrelu1'),
        SpatialConvolution(512,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch2'),
        ReLU(name='rrelu2'),
        SpatialConvolution(512,3,3, padding='SAME', name='rconv2'),], name='res5_2'),
    SpatialAveragePooling(7,7,1,1, name='pool5'),
    Affine(1000, name='fc6')
])