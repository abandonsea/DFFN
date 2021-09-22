# DFFN Architecture
Architecture used for "DFFN_indian_pines".
Convolution filter parameters per stage:
1. Number of filters: 16, Padding: 1, Kernel size: 3, Stride: 1
2. Number of filters: 32, Padding: 1, Kernel size: 3, Stride: 1 (except when stated otherwise)
3. Number of filters: 64, Padding: 1, Kernel size: 3, Stride: 1 (except when stated otherwise)

## Stage 1 (low level)
#### Res-block 1
- Conv-block 1 (Convolution1, BatchNorm1, Scale1)
- ReLU1
- Conv-block 2 (Convolution2, BatchNorm2, Scale2)
- ReLU2
- Conv-block 3 (Convolution3, BatchNorm3, Scale3)
- Eltwise1
- ReLU3
#### Res-block 2
- Conv-block 4 (Convolution4, BatchNorm4, Scale4)
- ReLU4
- Conv-block 5 (Convolution5, BatchNorm5, Scale5)
- Eltwise2
- ReLU5
#### Res-block 3
- Conv-block 6 (Convolution6, BatchNorm6, Scale6)
- ReLU6
- Conv-block 7 (Convolution7, BatchNorm7, Scale7)
- Eltwise3
- ReLU7
#### Res-block 4
- Conv-block 8 (Convolution8, BatchNorm8, Scale8)
- ReLU8
- Conv-block 9 (Convolution9, BatchNorm9, Scale9)
- Eltwise4
- ReLU9

## Stage 2 (mid level)
#### Res-block 5
- Conv-block 10 (Convolution10, BatchNorm10, Scale10)
  - Padding: 0, Kernel size: 1, Stride: 2
- Conv-block 11 (Convolution11, BatchNorm11, Scale11)
  - Stride: 2 
- ReLU10
- Conv-block 12 (Convolution12, BatchNorm12, Scale12)
- Eltwise5
- ReLU11
#### Res-block 6
- Conv-block 13 (Convolution13, BatchNorm13, Scale13)
- ReLU12
- Conv-block 14 (Convolution14, BatchNorm14, Scale14)
- Eltwise6
- ReLU13
#### Res-block 7
- Conv-block 15 (Convolution15, BatchNorm15, Scale15)
- ReLU14
- Conv-block 16 (Convolution16, BatchNorm16, Scale16)
- Eltwise7
- ReLU15
#### Res-block 8
- Conv-block 17 (Convolution17, BatchNorm17, Scale17)
- ReLU16
- Conv-block 18 (Convolution18, BatchNorm18, Scale18)
- Eltwise8
- ReLU17

## Stage 3 (high level)
#### Res-block 9
- Conv-block 19 (Convolution19, BatchNorm19, Scale19)
  - Padding: 0, Kernel size: 1, Stride: 2
- Conv-block 20 (Convolution20, BatchNorm20, Scale20)
  - Stride: 2 
- ReLU18
- Conv-block 21 (Convolution21, BatchNorm21, Scale21)
- Eltwise9
- ReLU19
#### Res-block 10
- Conv-block 22 (Convolution22, BatchNorm22, Scale22)
- ReLU20
- Conv-block 23 (Convolution23, BatchNorm23, Scale23)
- Eltwise10
- ReLU21
#### Res-block 11
- Conv-block 24 (Convolution24, BatchNorm24, Scale24)
- ReLU22
- Conv-block 25 (Convolution25, BatchNorm25, Scale25)
- Eltwise11
- ReLU23
#### Res-block 12
- Conv-block 26 (Convolution26, BatchNorm26, Scale26)
- ReLU24
- Conv-block 27 (Convolution27, BatchNorm27, Scale27)
- Eltwise12

## Dimension matching for stage 1
- Convolution_eltwise4
- BatchNorm_Convolution_eltwise4
- Scale_Convolution_eltwise4

## Dimension matching for stage 2
- Convolution_eltwise8
- BatchNorm_Convolution_eltwise8
- Scale_Convolution_eltwise8

## Fuse stages
- fuse1
- fuse2

## ReLu and pooling for fused data
- ReLU25
- Pooling1

## Fully connected
- InnerProduct1

## Softmax
- SoftmaxWithLoss1
