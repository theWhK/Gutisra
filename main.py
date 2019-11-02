import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# quebra a imagem em 5000 celulas, de 20x20 cada
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Numpy array - size (50,100,20,20)
x = np.array(cells)

# Training and testing train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE,train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)


# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy

# save the kNN Model
np.savez('knn_data.npz',train=train, train_labels=train_labels)
print "Saved"
