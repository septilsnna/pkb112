from mnist import MNIST
import numpy as np

mndata = MNIST('.data')
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()

images = np.array(images)
t_images = np.array(t_images)
labels = np.array(labels)
t_labels = np.array(t_labels)

#number of training dataset
len(labels)
#number of test dataset
len(t_labels)
#number of class
len(np.unique(labels))
#number of instances per class on training
for i in np.unique(labels):
    np.sum(labels==i)
#number of instances per class on test dataset
for i in np.unique(t_labels):
    np.sum(t_labels==i)
