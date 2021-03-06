# DFFN Architecture
Architecture used for "DFFN_indian_pines".
Convolution filter parameters per stage:
1. **Low level**: Number of filters: 16, Padding: 1, Kernel size: 3, Stride: 1
2. **Mid level**: Number of filters: 32, Padding: 1, Kernel size: 3, Stride: 1 (except when stated otherwise)
3. **High level**: Number of filters: 64, Padding: 1, Kernel size: 3, Stride: 1 (except when stated otherwise)

A _Conv-block_ is defined as: convolution, batch norm and scale.
The scale layer is a one-dimensional layer (array) with scale factors and a bias for each different band.

## Stage 1 (low level)
#### Res-block 1
- Conv-block 1
- ReLU1
- Conv-block 2
- ReLU2
- Conv-block 3
- Eltwise1 (sums ReLU1 and ConvBlock3)
- ReLU3
#### Res-block 2
- Conv-block 4
- ReLU4
- Conv-block 5
- Eltwise2 (sums ReLU3 and ConvBlock5)
- ReLU5
#### Res-block 3
- Conv-block 6
- ReLU6
- Conv-block 7
- Eltwise3 (sums ReLU5 and ConvBlock7)
- ReLU7
#### Res-block 4
- Conv-block 8
- ReLU8
- Conv-block 9
- Eltwise4 (sums ReLU7 and ConvBlock9)
- ReLU9
#### Res-block 5
- Conv-block 10
- ReLU10
- Conv-block 11
- Eltwise5 (sums ReLU9 and ConvBlock11)
- ReLU11

## Stage 2 (mid level)
#### Res-block 6
- Conv-block 12
  - Padding: 0, Kernel size: 1, Stride: 2
- Conv-block 13 (receives ReLU11)
  - Stride: 2 
- ReLU12
- Conv-block 14
- Eltwise6 (sums ConvBlock12 and ConvBlock14)
- ReLU13
#### Res-block 7
- Conv-block 15
- ReLU14
- Conv-block 16
- Eltwise7 (sums ReLU13 and ConvBlock16)
- ReLU15
#### Res-block 8
- Conv-block 17
- ReLU16
- Conv-block 18
- Eltwise8 (sums ReLU15 and ConvBlock18)
- ReLU17
#### Res-block 9
- Conv-block 19
- ReLU18
- Conv-block 20
- Eltwise9 (sums ReLU17 and ConvBlock20)
- ReLU19
#### Res-block 10
- Conv-block 21
- ReLU20
- Conv-block 22
- Eltwise10 (sums ReLU19 and ConvBlock22)
- ReLU21

## Stage 3 (high level)
#### Res-block 11
- Conv-block 23
  - Padding: 0, Kernel size: 1, Stride: 2
- Conv-block 24 (receives ReLU21)
  - Stride: 2 
- ReLU22
- Conv-block 25
- Eltwise11 (sums ConvBlock23 and ConvBlock25)
- ReLU23
#### Res-block 12
- Conv-block 26
- ReLU24
- Conv-block 27
- Eltwise12 (sums ReLU23 and ConvBlock27)
- ReLU25
#### Res-block 13
- Conv-block 28
- ReLU26
- Conv-block 29
- Eltwise13 (sums ReLU25 and ConvBlock29)
- ReLU27
#### Res-block 14
- Conv-block 30
- ReLU28
- Conv-block 31
- Eltwise14 (sums ReLU27 and ConvBlock31)
- ReLU29
#### Res-block 15
- Conv-block 32
- ReLU30
- Conv-block 33
- Eltwise15 (sums ReLU29 and ConvBlock33)

## Dimension matching and fusion
#### Dimension matching for stage 1
- Conv-block low-level (Eltwise5)
  - Num filters: 64, Padding: 1, Kernel size: 3, Stride: 4
#### Dimension matching for stage 2
- Conv-block mid-level (Eltwise10)
  - Num filters: 64, Padding: 1, Kernel size: 3, Stride: 2
#### Fuse stages
- fuse1 (low-level and mid-level)
- fuse2 (fuse1 and high-level)

## After fusion
#### ReLU and Pooling
- ReLU31
- Pooling1
#### Fully connected
- InnerProduct1
#### Softmax
- SoftmaxWithLoss1
