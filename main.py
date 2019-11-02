import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# quebra a imagem em 5000 celulas, de 20x20 cada
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Numpy array - tamanho (50,100,20,20)
x = np.array(cells)

# Training e testing.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Cria rótulos para os dados de teste e treino
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# inicia o knn, treina e testa para k=5
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE,train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)


# Checa a accuracy da classificação
# Comparando os resultados com test_labels e checa qual está errado
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy

# save the kNN Model
np.savez('knn_data.npz',train=train, train_labels=train_labels)
print "Saved"

