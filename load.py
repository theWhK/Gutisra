import numpy as np
import cv2
from matplotlib import pyplot as plt


#Carrega o modelo KNN
with np.load('knn_data.npz') as data:
    print data.files
    train = data['train']
    train_labels = data['train_labels']
 
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels) 
  
test_img=cv2.imread("imagens/8-4.png")
test_img =cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
test_img =cv2.resize(test_img, (20, 20))
x = np.array(test_img)
test_img = x.reshape(-1,400).astype(np.float32)
ret,result,neighbours,dist = knn.findNearest(test_img,k=5)

#Printa o predict
print result
print neighbours
print dist
