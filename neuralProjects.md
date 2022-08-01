# zacnem udemy
- install anaconda
https://www.anaconda.com/products/distribution

otvorim anaconda prompt
cd Downloads
create environment
conda env create -f name.yml

conda activate python_cvcourse

aktivuje enviroment a mame potrebne kniznice

//conda deactivate
na vypnutie

jupyter-lab


##
https://www.youtube.com/watch?v=qFJeN9V1ZsI
tetka tensorflow

vytvorim nove env
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

conda env create deeplizard
conda create --name deeplizard 

alebo presne specifikuj python
conda create --name py35 python=3.5


# To activate this environment, use
#
#     $ conda activate deeplizard
#
# To deactivate an active environment, use
#
#     $ conda deactivate


instalacia balikov
conda install numpy
conda install scikit-learn


tetka je nedoveryhodna preto idem radsej na nieco kde su aj zdrojaky


# ########################################################
https://www.youtube.com/watch?v=tpCFfeUEGs8&list=PL6vjgQ2-qJFfU2vF6-lG9DlSa4tROkzt9&index=1

https://github.com/mrdbourke/tensorflow-deep-learning

pouziva colaborator takze sa vraaciam spat k timovi

google collaboratory





vratil som sa k tetke

conda create --name tetka python=3.9
conda install numpy
conda install scikit-learn
conda install keras
conda install tensorflow

tetka.ipynb

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

use no kernel

ostranil som z User Path variable 4 pre anacondu

# pozeram ujka
https://www.youtube.com/watch?v=tpCFfeUEGs8&list=PL6vjgQ2-qJFfU2vF6-lG9DlSa4tROkzt9&index=4

staci cez pip nainstalovat tensorflow 2, tam je uz keras obsiahnuty
https://www.activestate.com/resources/quick-reads/how-to-install-keras-and-tensorflow/

v colabe ked kliknem ctrl+shift+space tak mi ukaze napovedu


#
dot product
https://www.youtube.com/watch?v=0iNrGpwZwog

cross product
https://www.youtube.com/watch?v=gPnWm-IXoAY


# bourke 
3:00:40 tu som skoncil s basic tensor operations

Keras reading captcha
https://www.youtube.com/watch?v=SHo3hbsJs_U

# tutorial na captche pylessons.com
https://pylessons.com/TensorFlow-CAPTCHA-solver-introduction
https://www.youtube.com/watch?v=4D5RN2yKlG4
https://www.youtube.com/watch?v=NScV771cC4U&list=PLbMO9c_jUD456277j2fHAUip19xfxIAx0&index=2
https://www.youtube.com/watch?v=OS5GDGU-jvc

TFRecord file
https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c



# recurent neural networks
https://www.youtube.com/watch?v=qjrad0V0uJE

# convolutional neural network
https://www.youtube.com/watch?v=YRhxdVk_sIs
- obsahuje convolution layers
- are able to detect patterns
- filters detect patterns = edge detector filter
filter is small matrix 3x3, it convolve through input, take dot product
https://course17.fast.ai/lessons/lesson4.html



# bourke
numpy and tensorflow
difference between list and numpy array
https://python.plainenglish.io/python-list-vs-numpy-array-whats-the-difference-7308cd4b52f6

# bourke nn regression
3:51:50
- one hot encoding tensor
- example of supervised learning
4"06:18

- book Hands-on Machine Learning with Scikit-learn Aurelien Geron

4:15:53 start coding regression model

5:04:27 improving regression model

5:16:55 Evaluating a model


loss function = cost fucntion - erro function
error between output and target value
https://www.youtube.com/watch?v=-qT8fJTP3Ks

loss function for classification problems
Cross entropy
https://www.youtube.com/watch?v=Md4b67HvmRo

optimizers?
- how to change the line


Visualizing prediction 
6:00:00
