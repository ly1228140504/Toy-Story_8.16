from ultralytics import YOLO
import numpy as np
import cv2
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import lowrank_cons
import concurrent.futures
import os
import sys
sys.path.append("./image_to_signal")
from spectrum import process_a_image
from kmeans import image_to_signal_real
threshold_values = {}
h = [1]


def process_image(image, k=1.5):
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    U, s, Vt = np.linalg.svd(image)
    # print(f"Image: {os.path.basename(image_path)} - s[0]: {s[0]}, s[-1]: {s[-1]}")
    s = s * k
    # print(f"Image: {os.path.basename(image_path)} - s[0]/s[-1]: {s[0] / s[-1]}")

    S = np.diag(s)
    zerosmatrix = np.zeros((s.shape[0], Vt.shape[0] - s.shape[0]))
    S = np.hstack((S, zerosmatrix))
    F = np.dot(np.dot(U, S), Vt)
    return F
    # output_path = os.path.join(output_folder, os.path.basename(image_path))
    # cv2.imwrite(output_path, F)
    # print(f"Saved processed image to {output_path}")


def Hist(img):
   row, col = img.shape
   y = np.zeros(256)
   for i in range(0, row):
      for j in range(0, col):
         y[img[i, j]] += 1
   # x = np.arange(0, 256)
   # plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   # plt.show()
   return y

def regenerate_img(img, threshold):
    row, col = img.shape
    y = np.zeros((row, col))
    for i in range(0, row):
        for j in range(0, col):
            if img[i, j] >= threshold:
                y[i, j] = 255
            else:
                y[i, j] = 0
    return y

def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i] > 0:
            cnt += h[i]
    return cnt

def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w

def mean(s, e):
    m = 0
    w = wieght(s, e)
    if w == 0:  # 添加检查分母是否为零的代码
        return 0
    for i in range(s, e):
        m += h[i] * i
    return m / float(w)

def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    if w == 0:  # 添加检查分母是否为零的代码
        return 0
    for i in range(s, e):
        v += ((i - m) ** 2) * h[i]
    v /= w
    return v

def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)
        mb = mean(0, i)

        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)
        mf = mean(i, len(h))

        V2w = wb * (vb) * (vb) + wf * (vf) * (vf)
        V2b = wb * wf * (mb - mf) ** 2

        fw = open("trace.txt", "a")
        fw.write('T=' + str(i) + "\n")

        fw.write('Wb=' + str(wb) + "\n")
        fw.write('Mb=' + str(mb) + "\n")
        fw.write('Vb=' + str(vb) + "\n")

        fw.write('Wf=' + str(wf) + "\n")
        fw.write('Mf=' + str(mf) + "\n")
        fw.write('Vf=' + str(vf) + "\n")

        fw.write('within class variance=' + str(V2w) + "\n")
        fw.write('between class variance=' + str(V2b) + "\n")
        fw.write("\n")

        if not math.isnan(V2w):
            threshold_values[i] = V2w

def get_optimal_threshold():
    min_V2w = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    print('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


def del_txt(gray_image,categories,targets):
    image_height, image_width = gray_image.shape

    # 遍历所有目标
    for i in range(targets.shape[0]):
        # 如果类别为2
        if categories[i] == 2:
            # 获取中心坐标和宽高
            x_center, y_center, width, height = targets[i]

            # 计算边界框
            x1 = max(0, int(x_center - width // 2))
            y1 = max(0, int(y_center - height // 2))
            x2 = min(image_width, int(x_center + width // 2))
            y2 = min(image_height, int(y_center + height // 2))

            # 将边界框内的像素值设置为255
            gray_image[y1:y2, x1:x2] = 255
    return gray_image



def del_txt1(gray_image,categories,targets):
    image_height, image_width = gray_image.shape

    # 遍历所有目标
    for i in range(targets.shape[0]):
        # 如果类别为2
        if categories[i] == 0:
            # 获取中心坐标和宽高
            x_center, y_center, width, height = targets[i]

            # 计算边界框
            x1 = max(0, int(x_center - width // 2))
            y1 = max(0, int(y_center - height // 2))
            x2 = min(image_width, int(x_center + width // 2))
            y2 = min(image_height, int(y_center + height // 2))

            # 将边界框内的像素值设置为255
            ck = gray_image[y1:y2, x1:x2]
            break
    return ck

def save_target_images(gray_image, categories, targets, base_filename):
    image_height, image_width = gray_image.shape
    count = 1
    targets_sort = sort_and_label_targets(categories,targets)
    # 遍历所有目标
    for i in range(len(targets_sort)):
        # 如果类别为0

        x_center, y_center, width, height = targets_sort[i]

        # 计算边界框
        x1 = max(0, int(x_center - width // 2))
        y1 = max(0, int(y_center - height // 2))
        x2 = min(image_width, int(x_center + width // 2))
        y2 = min(image_height, int(y_center + height // 2))

        # 提取边界框内的图像区域
        target_region = gray_image[y1:y2, x1:x2]

        # 构造文件名并保存图片
        filename = f"{base_filename}_{count}.jpg"
        cv2.imwrite(filename, target_region)
        count += 1

    print(f"Saved {count-1} images.")


def sort_and_label_targets(categories, targets):
    category_0_targets = []
    for i in range(targets.shape[0]):
        if categories[i] == 0:
            category_0_targets.append(targets[i].cpu())

    # 转换为 numpy 数组
    category_0_targets = np.array(category_0_targets)

    # 根据 x 值排序
    sorted_targets = sorted(category_0_targets, key=lambda x: x[0])

    # 分组并命名
    labeled_targets = []
    group_size = 3  # 每组的大小
    for group_start in range(0, len(sorted_targets), group_size):
        # 取出一组
        group = sorted_targets[group_start:group_start + group_size]

        # 根据 y 值从小到大排序
        group_sorted_by_y = sorted(group, key=lambda x: x[1])

        # 给每个目标命名
        for j, target in enumerate(group_sorted_by_y):
            labeled_targets.append(target)
    return labeled_targets






def signal_restruct_fold(input_folder):
    # input_folder = r'E:\code_crazy\pnet2024-1\pnet2024\final+process\test'
    model =  YOLO('best.pt')#
    results = model.predict(source=input_folder, save=False, save_txt=False)
    # print(results.boxes.xywh)
    output = []
    for idx, filename in enumerate(os.listdir(input_folder)):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)

        image = process_a_image(input_path)
        # image = cv2.resize(image, (1700, 1500))
        aa = idx
        sort_and_label_targets(results[aa].boxes.cls, results[aa].boxes.xywh)
        processed_image = del_txt(image, results[aa].boxes.cls, results[aa].boxes.xywh)

        op_thres = 190
        res = regenerate_img(processed_image, op_thres)
        img = 255 - res

        categories = results[aa].boxes.cls
        targets = results[aa].boxes.xywh
        gray_image = img

        image_height, image_width = gray_image.shape
        count = 1
        targets_sort = sort_and_label_targets(categories, targets)
        # 遍历所有目标
        output_list = []
        for i in range(len(targets_sort)):

            x_center, y_center, width, height = targets_sort[i]

            # 计算边界框
            x1 = max(0, int(x_center - width // 2))
            y1 = max(0, int(y_center - height // 2))
            x2 = min(image_width, int(x_center + width // 2))
            y2 = min(image_height, int(y_center + height // 2))

            # 提取边界框内的图像区域
            target_region = gray_image[y1:y2, x1:x2]
            output_list.append(image_to_signal_real(target_region))
        output.append(np.array(output_list))
    return output


def expand_array_with_nan(array, target_cols):
    original_rows, original_cols = array.shape
    if target_cols <= original_cols:
        return array

    # 创建新数组并填充 NaN
    expanded_array = np.full((original_rows, target_cols), np.nan)

    # 复制原始数据
    expanded_array[:, :original_cols] = array

    return expanded_array

def get_fin(data,chan,num):
    if data.shape[0] >= chan:
        data = data[:chan,:]
    else:
        rows_to_add = chan - data.shape[0]

    # 取得原始数组的最后一行
        last_row = data[-1, :]

    # 扩展数组
        data = np.vstack([data] + [last_row] * rows_to_add)
    data_nan = expand_array_with_nan(data, num)

    return data,data_nan

def resize_image(image_path, output_path, size=(1700, 1500)):
    # 打开图片
    img = cv2.imread(image_path)

    # 调整图片大小
    resized_img = cv2.resize(img, size)

    # 保存调整后的图片到指定路径
    cv2.imwrite(output_path, resized_img)
"""
from PIL import Image
def resize_image(image_path, output_path, size=(1700, 1500)):
    with Image.open(image_path) as img: # 调整图片大小 resized_img = img.resize(size) # 保存调整后的图片到指定路径 resized_img.save(output_path
        resized_img = img.resize(size)  # 保存调整后的图片到指定路径
        #print(resized_img.size)
        resized_img.save(output_path)
"""

def signal_restruct(image_path,num_samples,num_signals):
    # input_folder = r'E:\code_crazy\pnet2024-1\pnet2024\final+process\test'
    #sys.path.append("./image_to_signal/")
    W = cv2.imread(image_path).shape[1]
    H = cv2.imread(image_path).shape[0]
    W_bili = 1700/W
    H_bili = 1500/H
    #print(f"W,H,W_bili,H_bili:{W} {H} {W_bili} {H_bili}")
    resize_image(image_path, image_path, size=(1700, 1500))
    model = YOLO('./image_to_signal/best.pt')#
    results = model.predict(source=image_path, save=False, save_txt=False)
    # print(results.boxes.xywh)
    output = []
    # 构建完整的文件路径
    input_path = image_path

    image = process_a_image(input_path)
    # image = cv2.resize(image, (1700, 1500))
    #print(img.shape)
    aa = 0
    sort_and_label_targets(results[aa].boxes.cls, results[aa].boxes.xywh)
    processed_image = del_txt(image, results[aa].boxes.cls, results[aa].boxes.xywh)

    op_thres = 190*2.5
    res = regenerate_img(processed_image, op_thres)
    img = 255 - res

    categories = results[aa].boxes.cls
    targets = results[aa].boxes.xywh
    gray_image = img
    #print(f"gray_image.shape{gray_image.shape}")

    image_height, image_width = gray_image.shape
    count = 1
    targets_sort = sort_and_label_targets(categories, targets)
    #print(f"targets_sort{targets_sort}")
    # 遍历所有目标
    output_list = []
    output_list_250 = []
    if len(targets_sort) == 0:
        out_nan = np.zeros(num_signals,num_samples)
        out_250 = np.zeros(12,250)
    else:
        for i in range(len(targets_sort)):

            x_center, y_center, width, height = targets_sort[i]
            x_center = x_center*W_bili
            width = width*W_bili
            y_center = y_center*H_bili
            height = height*H_bili


            # 计算边界框
            x1 = max(0, int(x_center - width // 2))
            y1 = max(0, int(y_center - height // 2))
            x2 = min(image_width, int(x_center + width // 2))
            y2 = min(image_height, int(y_center + height // 2))

            #print(f"x1:{x1}")
            #print(f"x2:{x2}")
            #print(f"y1:{y1}")
            #print(f"y2:{y2}")

            # 提取边界框内的图像区域
            target_region = gray_image[y1:y2, x1:x2]
            output_list.append(image_to_signal_real(target_region,num_samples//4))
            output_list_250.append(image_to_signal_real(target_region, 250))
        output_list = np.array(output_list)
        output_list_250 = np.array(output_list_250)
        # get_fin(output_list, 12, 250)
        _,out_nan = get_fin(output_list,num_signals,num_samples)
        out_250,_ = get_fin(output_list_250, 12, 250)
    return out_250,out_nan



# a = r'E:\code_crazy\pnet2024-1\pnet2024\final+process\test\10000_lr-0.jpg'
# b1,b2 = signal_restruct(a,1200,12)
#
# c = 1