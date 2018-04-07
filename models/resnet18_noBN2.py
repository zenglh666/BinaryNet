from nnUtils import *

model = Sequential([
    SpatialConvolution(64,7,7,2,2, padding='SAME', name='conv1', bias=False),
    ReLU(name='relu1'),
    SpatialMaxPooling(3,3,2,2, name='pool1', padding='SAME'),
    Residual([
        SpatialConvolution(64,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res2_1'),
    Residual([
        SpatialConvolution(64,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res2_2'),
    Residual([
        SpatialConvolution(64,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res2_3'),
    Residual([
        SpatialConvolution(64,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res2_4'),
    Residual([
        SpatialConvolution(128,3,3,2,2, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], connect=False, name='res3_1'),
    Residual([
        SpatialConvolution(128,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res3_2'),
    Residual([
        SpatialConvolution(128,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res3_3'),
    Residual([
        SpatialConvolution(128,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res3_4'),
    Residual([
        SpatialConvolution(256,3,3,2,2, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], connect=False, name='res4_1'),
    Residual([
        SpatialConvolution(256,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res4_2'),
    Residual([
        SpatialConvolution(256,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res4_3'),
    Residual([
        SpatialConvolution(256,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res4_4'),
    Residual([
        SpatialConvolution(512,3,3,2,2, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], connect=False, name='res5_1'),
    Residual([
        SpatialConvolution(512,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res5_2'),
    Residual([
        SpatialConvolution(512,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res5_3'),
    Residual([
        SpatialConvolution(512,3,3, padding='SAME', name='rconv1', bias=False),
        ReLU(name='rrelu1'),
        ], name='res5_4'),
    SpatialAveragePooling(7,7,1,1, name='pool5'),
    Affine(1000, name='fc6')
])