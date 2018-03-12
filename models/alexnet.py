from nnUtils import *

model = Sequential([
    SpatialConvolution(96,11,11,4,4, padding='SAME'),
    ReLU(),
    LocalResposeNormalize(2, 1e-05, 0.75),
    SpatialMaxPooling(3,3,2,2),
    SpatialConvolution(256,5,5, padding='SAME'),
    ReLU(),
    SpatialMaxPooling(3,3,2,2),
    LocalResposeNormalize(2, 1e-05, 0.75),
    SpatialConvolution(384,3,3, padding='SAME'),
    ReLU(),
    SpatialConvolution(384,3,3, padding='SAME'),
    ReLU(),
    SpatialConvolution(256,3,3, padding='SAME'),
    SpatialMaxPooling(3,3,2,2),
    ReLU(),
    Affine(4096),
    ReLU(),
    Dropout(0.5),
    Affine(4096),
    ReLU(),
    Dropout(0.5),
    Affine(1000)
])
