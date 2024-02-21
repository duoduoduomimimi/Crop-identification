#限制对比度自适应直方图均衡化
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os,shutil

# img = Image.open('AllData/test/banana/image_62.jpg').convert('RGB')
img = Image.open('putao2.jpg').convert('RGB')
img = np.uint8(img)

imgr = img[:,:,0]
imgg = img[:,:,1]
imgb = img[:,:,2]

claher = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
claheg = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
claheb = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
cllr = claher.apply(imgr)
cllg = claheg.apply(imgg)
cllb = claheb.apply(imgb)
rgb_img = np.dstack((cllr,cllg,cllb))
claher2 = cv2.createCLAHE(tileGridSize=(4,4))
claheg2 = cv2.createCLAHE(tileGridSize=(4,4))
claheb2 = cv2.createCLAHE(tileGridSize=(4,4))
cllr2 = claher2.apply(imgr)
cllg2 = claheg2.apply(imgg)
cllb2 = claheb2.apply(imgb)
rgb_img2 = np.dstack((cllr2,cllg2,cllb2))
cv2.imwrite(os.path.join('result', "a1" + ".jpg"), rgb_img)
plt.subplot(1,3,1),plt.imshow(img)
plt.title('orig'),plt.axis('off')
plt.subplot(1,3,2),plt.imshow(rgb_img2)
plt.title('ahe'),plt.axis('off')
plt.subplot(1,3,3),plt.imshow(rgb_img)
plt.title('Clahe'),plt.axis('off')
plt.show()
plt.subplot(1,1,1),plt.imshow(rgb_img)
plt.title('Clahe'),plt.axis('off')
plt.show()
