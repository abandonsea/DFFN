# DFFN
This is an implementation of the DFFN network presented in the paper "Hyperspectral Image Classification With Deep Feature Fusion Network".
The original implementation of the network is in Matlab, using the Caffe Framework and is available in a [GitHub repository](https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018).

# Tested with
Python 3.9

Pytorch 1.9.0  

CPU or GPU

# Run the DFFN
Please set your parameters in train.py or test.py before running them. 

To train, run:
```python
# Trains network multiple times (see parameters in file)
python train.py
``` 

To test, run:
```python
# Not yet fully implemented
python test.py
```

# About datasets and checkpoints
The datasets are available [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).
The used datasets for this implementation are: PaviaU, Indian Pines and Salinas.

# Authorship disclaimer
While I did write/review/modify alone the entire code, many parts of the code are heavily based on the original Matlab implementation, as well as the [S-DMM implementation](https://github.com/ShuGuoJ/S-DMM).
