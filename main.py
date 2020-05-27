from model import *
from preprocess import *
import matplotlib.pyplot as plt


size = (224, 224)
model = model((size[0], size[1], 3))

train = read_data('../input/segment/Data/Training_Images', size, 1)
res = read_data('../input/segment/Data/Ground_Truth', size, 0)
print(train.shape)
print(res.shape)

model.fit(train, res, batch_size=16, epochs=30)
model.save('trained_model1.h5')
# model.load_weights('../input/model-trained/trained_model')

k = 93
img = train[k]
r = res[k]
plt.imshow(img.reshape((size[0], size[1], 3)), cmap='gray')

plt.imshow(r.reshape(size), cmap='gray')

p = model.predict(img.reshape((1, size[0], size[1], 3)))
p = p.reshape(size)

print(p.tolist())

a = .5
p[p > a] = 1
p[p <= a] = 0

plt.imshow(p, cmap='gray')
cv2.imwrite("res1.jpg", p)