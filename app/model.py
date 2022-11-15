from keras.layers import Dense, Input, concatenate
from keras.models import Model, Sequencial

model = Sequencial()

inDense = Input(shape=(4, ), dtype='int32', name='input-4')
leftDense = Dense(10, name='left-model')(inDense)
rightDense = Dense(16, name='right-model')(inDense)
mergedDense = concatenate([leftDense, rightDense], name='merge')
outDense = Dense(2, name='final')(mergedDense)
model2 = Model(inDense, outDense)