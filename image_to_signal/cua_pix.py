import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
image_dir = r'E:\code_crazy\pnet2024-1\pnet2024\yolo\image3_3'
label_dir = 'D:/npy'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

label_dir = label_dir
pix = []
sig = []
for idx in range(len(image_files)):
    # idx = 1
    img_name = image_files[idx]
    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path).convert('L')

    original_height = image.size[1]

    label_name = img_name.replace('.jpg', '.npy')
    first_dash_index = label_name.find('_')  # 找到第一个 '-'
    if first_dash_index != -1:
        second_dash_index = label_name.find('_', first_dash_index + 1)  # 找到第二个 '-'
        if second_dash_index != -1:
            label_name = (label_name[:second_dash_index] + '-' +
                          label_name[second_dash_index + 1:])

    # label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.npy'))
    label_path = os.path.join(label_dir, label_name)
    label = np.load(label_path)
    label = label[~np.isnan(label)]
    label_range = label.max() - label.min()
    sig.append(label_range)
    pix.append(original_height )


sig = np.array(sig)
pix = np.array(pix)
a = sig/pix
k = np.median(a)
plt.plot(a)
k = 0.016789473684210528
plt.show()
k = 1
