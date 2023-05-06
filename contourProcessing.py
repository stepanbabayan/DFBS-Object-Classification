import random
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

images = []
for image in os.listdir('data/extractedContours'):
    images.append(os.path.join('data/extractedContours', image))

plt.figure(1, figsize=(15, 9))
plt.axis('off')
n = 0
for i in range(16):
  n += 1
  random_img = random.choice(images)
  imgs = cv2.imread(random_img)
  plt.subplot(4, 4, n)
  plt.axis('off')
  plt.imshow(imgs)

plt.show()
