from nnUtils import *

model = Sequential([
    AccurateBinarizedWeightOnlySpatialConvolution(64,7,7,2,2, padding='SAME', name='conv1'),
    BatchNormalization(name='batch1'),
    HardTanh(name='HardTanh1'),
    SpatialMaxPooling(3,3,2,2, name='pool1', padding='SAME'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res2_1'),
    HardTanh(name='rHardTanh2'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(64,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res2_2'),
    HardTanh(name='rHardTanh2'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(128,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], connect=False, name='res3_1'),
    HardTanh(name='rHardTanh2'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res3_2'),
    HardTanh(name='rHardTanh2'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(256,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], connect=False, name='res4_1'),
    HardTanh(name='rHardTanh2'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res4_2'),
    HardTanh(name='rHardTanh2'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(512,3,3,2,2, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], connect=False, name='res5_1'),
    HardTanh(name='rHardTanh2'),
    Residual([
        AccurateBinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv1'),
        BatchNormalization(name='rbatch1'),
        HardTanh(name='rHardTanh1'),
        AccurateBinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', name='rconv2'),
        BatchNormalization(name='rbatch2'),], name='res5_2'),
    HardTanh(name='rHardTanh2'),
    SpatialAveragePooling(7,7,1,1, name='pool5'),
    BinarizedWeightOnlyAffine(1000, name='fc6')
])