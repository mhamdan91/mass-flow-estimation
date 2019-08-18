import matplotlib.pyplot as plt
import numpy as np
import os
img_path = '/home/moe/Desktop/bambo_Logs/Data/sample_dataset/testing/2015_07_21_1203_bamb_70_630'
dir_img = sorted(os.listdir(img_path))
fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 3))
t2 = ax2.text(len(dir_img)-150, 7, 'Prediction: {0:2f}'.format(np.sqrt(np.sqrt(0))))
for i, img in enumerate(dir_img):
    imgx = os.path.join(img_path, img)
    im = plt.imread(imgx)
    ax.imshow(im)
    ax1.imshow(im)
    ax2.axis([0, len(dir_img), 0, 8])
    ax2.plot(i, np.sqrt(np.sqrt(i)), marker='.')
    t2.set_text('Prediction: {0:2f}'.format(np.sqrt(np.sqrt(i))))
    # ax2.legend(loc='upper_left')
    plt.pause(.1333)
    plt.draw()
