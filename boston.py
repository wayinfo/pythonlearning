from keras.datasets import boston_housing
import os
import tensorflow as tf
import time

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
# 限制显存用2GB,还可以在只有单GPU的环境模拟多GPU进行调试,代码就在GPU:0上建立了两个显存均为 2GB 的虚拟 GPU
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
# )

# 设置仅在需要时申请显存空间
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# time.sleep(3600)
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

import numpy as np 

k= 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
starttime = time.time()
with tf.device('/gpu:0'):
    for i  in range(k):
        print('processing fold #',i)
        # 验证集
        val_data = train_data[i * num_val_samples : (i+1)*num_val_samples]
        val_targets = train_targets[i * num_val_samples : (i+1)*num_val_samples]
        # 训练集
        partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
        partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)

        model = build_model()
        model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=10,verbose=0)

        val_mse , val_mae = model.evaluate(val_data,val_targets,verbose=0)
        all_scores.append(val_mae)

print("total take time is {0}".format(time.time()-starttime))
print(np.array(all_scores).mean())

