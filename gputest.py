import tensorflow as tf
print(tf.__version__)
#print(tf.test.is_gpu_available())
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))