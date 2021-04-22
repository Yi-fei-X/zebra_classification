import os
import numpy as np
from skimage import io
from skimage import feature as ft
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load segmented data
path = "./Zebra_Alexnet/Zebra_crop/"
str = "./Zebra_Alexnet/Zebra_crop/*.jpg"
dir = os.listdir(path)
size = 1194
label = [None]*size
for i in range(size):
    label[i] = dir[i].split('_')[0] + '_' + dir[i].split('_')[1] + '_' + dir[i].split('_')[2]

coll = io.ImageCollection(str)
coll_array = np.array(coll)

# HOG feature extractor
new_coll = np.zeros([size, 130536])

for i in range(size):
    new_coll[i,:] = ft.hog(coll_array[i],
                       orientations=9,
                       pixels_per_cell=(8,8),
                       cells_per_block=(2,2),
                       block_norm='L2',
                       visualize=False,
                       transform_sqrt=True,
                       feature_vector=True)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(new_coll, label, test_size=0.30)

# SVM classifier
SVM = svm.LinearSVC()
SVM.fit(X_train, y_train)
y_pred_test = SVM.predict(X_test)
y_pred_train = SVM.predict(X_train)

# Training accuracy
correct = 0
total = len(y_train)
for i in range(len(y_train)):
    if y_pred_train[i] == y_train[i]:
        correct = correct + 1
print('Accuracy train: %d %%' % (
    100 * correct / total))

# Test accuracy
correct = 0
total = len(y_test)
for i in range(len(y_test)):
    if y_pred_test[i] == y_test[i]:
        correct = correct + 1
print('Accuracy test: %d %%' % (
    100 * correct / total))
