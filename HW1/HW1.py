import matplotlib.pyplot as plt
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for a in range(12):
    plt.subplot(4,3,a+1)
    plt.imshow(xtrain[a])
    plt.xlabel(f"Data Point {a}: " + class_labels[int(ytrain[a])])



print(f"Pre-shaped training data: {xtrain.shape}")
print(f"Pre-shaped testing data: {xtest.shape}")
print(f"Flattened training data: {xtrain.reshape([xtrain.shape[0], -1]).shape}")
print(f"Flattened testing data: {xtest.reshape([xtest.shape[0], -1]).shape}")

plt.subplots_adjust(hspace=1.0, wspace=0.5)
plt.show()