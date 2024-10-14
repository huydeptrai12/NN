import sys
sys.path.append("C:\\Users\\nguye\\OneDrive\\Desktop\\CNN\\EpyNN")

import random

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application/library specific imports
import epynn.initialize
from epynn.commons.maths import relu, softmax
from epynn.commons.library import (
    configure_directory,
    read_model,
)
from epynn.network.models import EpyNN
from epynn.embedding.models import Embedding
from epynn.convolution.models import Convolution
from epynn.pooling.models import Pooling
from epynn.flatten.models import Flatten
from epynn.dropout.models import Dropout
from epynn.dense.models import Dense
from epynnlive.captcha_mnist.settings import se_hPars

from epynnlive.dummy_image.prepare_dataset import load_mnist_data

########################## CONFIGURE ##########################
random.seed(1)
np.random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')

configure_directory()

X_train, y_train, X_test, y_test = load_mnist_data(limit=10000)
# print(len(X_features))
# print(X_features[0].shape)
# print(set(X_features[0].flatten().tolist()))  
# print(len(Y_label))

embedding = Embedding(X_data=X_train,
                      Y_data=y_train,
                      X_scale=True,
                      Y_encode=True,
                      batch_size=5,
                      relative_size=(1, 0, 0))

name = 'Flatten_Dropout02_Dense-64-relu_Dropout05_Dense-10-softmax'

se_hPars['learning_rate'] = 0.01
se_hPars['softmax_temperature'] = 5

flatten = Flatten()

dropout1 = Dropout(drop_prob=0.2)

hidden_dense = Dense(64, relu)

dropout2 = Dropout(drop_prob=0.5)

dense = Dense(10, softmax)

layers = [embedding, flatten, dropout1, hidden_dense, dropout2, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='CCE', seed=1, se_hPars=se_hPars.copy(), end='\n')
model.train(epochs=5, init_logs=True)

#model.plot(path=False) 
#model.write()


predicted = model.predict(X_test, X_scale=True).P
print(predicted)
print(y_test)
x = (predicted == y_test)
print(x.sum())