from mnist import MNIST
import numpy as np
from lib import *

mndata = MNIST('.data')
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()

images = np.array(images)
t_images = np.array(t_images)
labels = np.array(labels)
t_labels = np.array(t_labels)

print_statistics(images, t_images, labels, t_labels)

bias = np.ones((images.shape,1))
t_bias = np.ones((t_images.shape,1))

np.hstack(bias, images)
np.hstack(t_bias, t_images)