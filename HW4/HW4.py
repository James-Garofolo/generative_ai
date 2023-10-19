import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.fashion_mnist.load_data()

xtrain = xtrain.reshape(-1, 28, 28, 1).astype('float32')/255.0
xtest = xtest.reshape(-1, 28, 28, 1).astype('float32')/255.0

xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.1)

print(tf.reduce_max(xtrain), xtrain.shape)
ytrain = tf.one_hot(ytrain, 10)
yval = tf.one_hot(yval, 10)
ytest = tf.one_hot(ytest, 10)



inputs = tf.keras.Input(shape=(28, 28, 1), name="img")
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
max = tf.keras.layers.MaxPooling2D(2)(x)
avg = tf.keras.layers.AvgPool2D(2)(x)
block_1_output = tf.keras.layers.Concatenate()([max, avg])

x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = tf.keras.layers.add([x, block_1_output])
max = tf.keras.layers.MaxPooling2D(2)(x)
avg = tf.keras.layers.AvgPool2D(2)(x)
block_2_output = tf.keras.layers.Concatenate()([max, avg])

x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(block_2_output)
x = tf.keras.layers.add([x, block_2_output])
max = tf.keras.layers.MaxPooling2D(2)(x)
avg = tf.keras.layers.AvgPool2D(2)(x)
block_3_output = tf.keras.layers.Concatenate()([max, avg])

x = tf.keras.layers.Flatten()(block_3_output)

x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="toy_resnet")
model.summary()


learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=5000,
    decay_rate=0.96)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), loss="categorical_crossentropy", metrics=['accuracy'])
#tf.keras.utils.plot_model(model, to_file="hw_cnn.png", show_shapes=True, show_layer_names=True)

print(model.layers[1].get_weights()[0].shape)

history = model.fit(xtrain, ytrain, 64, 50, validation_data=(xval, yval))

model.evaluate(xtest, ytest)

kernel_max = tf.reduce_max(tf.abs(model.layers[1].get_weights()[0]), axis=None)
kernels = []
for a in range(6):
    kernels.append(model.layers[1].get_weights()[0][:,:,:,a]/kernel_max)

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss (a.u.)')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.figure(3)
plt.subplot(3,2,1)
plt.imshow(model.layers[1].get_weights()[0][:,:,:,0], cmap='gray')
plt.xlabel("kernel 0")
plt.subplot(3,2,2)
plt.imshow(model.layers[1].get_weights()[0][:,:,:,1], cmap='gray')
plt.xlabel("kernel 1")
plt.subplot(3,2,3)
plt.imshow(model.layers[1].get_weights()[0][:,:,:,2], cmap='gray')
plt.xlabel("kernel 2")
plt.subplot(3,2,4)
plt.imshow(model.layers[1].get_weights()[0][:,:,:,3], cmap='gray')
plt.xlabel("kernel 3")
plt.subplot(3,2,5)
plt.imshow(model.layers[1].get_weights()[0][:,:,:,4], cmap='gray')
plt.xlabel("kernel 4")
plt.subplot(3,2,6)
plt.imshow(model.layers[1].get_weights()[0][:,:,:,5], cmap='gray', vmin=-1.0, vmax=1.0)
plt.xlabel("kernel 5")
plt.subplots_adjust(hspace=1.0, wspace=0.5)
plt.show()
