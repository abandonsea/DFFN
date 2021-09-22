DFFN_indian_pines
- DATA

# STAGE 1
## Resblock 1
- Convolution1
- BatchNorm1
- Scale
- ReLU1
- Convolution2
- BatchNorm2
- Scale2
- ReLU2
- Convolution3
- BatchNorm3
- Scale3
- Eltwise1
- ReLU3
- Convolution4
- BatchNorm4
- Scale4
- ReLU4
- Convolution5
- BatchNorm5
- Scale5
- Eltwise2
- ReLU5
- Convolution6
- BatchNorm6
- Scale6
- ReLU6
- Convolution7
- BatchNorm7
- Scale7
- Eltwise3
- ReLU7

- Convolution8
- BatchNorm8
- Scale8
- ReLU8
- Convolution9
- BatchNorm9
- Scale9
- Eltwise4
- ReLU9

- Convolution10
- BatchNorm10
- Scale10
- Convolution11
- BatchNorm11
- Scale11
- ReLU10
- Convolution12
- BatchNorm12
- Scale12
- Eltwise5
- ReLU11
- Convolution13
- BatchNorm13
- Scale13
- ReLU12
- Convolution14
- BatchNorm14
- Scale14
- Eltwise6
- ReLU13
- Convolution15
- BatchNorm15
- Scale15
- ReLU14
- Convolution16
- BatchNorm16
- Scale16
- Eltwise7
- ReLU15
- Convolution17
- BatchNorm17
- Scale17
- ReLU16
- Convolution18
- BatchNorm18
- Scale18
- Eltwise8
- ReLU17
- Convolution19
- BatchNorm19
- Scale19
- Convolution20
- BatchNorm20
- Scale20
- ReLU18
- Convolution21
- BatchNorm21
- Scale21
- Eltwise9
- ReLU19
- Convolution22
- BatchNorm22
- Scale22
- ReLU20
- Convolution23
- BatchNorm23
- Scale23
- Eltwise10
- ReLU21
- Convolution24
- BatchNorm24
- Scale24
- ReLU22
- Convolution25
- BatchNorm25
- Scale25
- Eltwise11
- ReLU23
- Convolution26
- BatchNorm26
- Scale26
- ReLU24
- Convolution27
- BatchNorm27
- Scale27
- Eltwise12


- Convolution_eltwise4
- BatchNorm_Convolution_eltwise4
- Scale_Convolution_eltwise4


- Convolution_eltwise8
- BatchNorm_Convolution_eltwise8
- Scale_Convolution_eltwise8

- fuse1
- fuse2

- ReLU25

- Pooling1


- InnerProduct1
- SoftmaxWithLoss1

- CHECK_ACCURACY