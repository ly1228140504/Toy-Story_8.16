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

# def regenerate_img(img, threshold):
# #     row, col = img.shape
# #     y = np.zeros((row, col))
# #     for i in range(0, row):
# #         for j in range(0, col):
# #             if img[i, j] >= threshold:
# #                 y[i, j] = 255
# #             else:
# #                 y[i, j] = 0
# #     return y
def regenerate_img(img, threshold):
    # 使用矢量化操作来替代循环
    y = np.where(img >= threshold, 255, 0)
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




import time


input_folder = r'E:\code_crazy\pnet2024-1\pnet2024\final+process\test'
model =  YOLO('best.pt')#
results = model.predict(source=input_folder, save=False, save_txt=False)
# print(results.boxes.xywh)
output = []
for idx, filename in enumerate(os.listdir(input_folder)):
    # 构建完整的文件路径
    input_path = os.path.join(input_folder, filename)

    # 测量 process_a_image 的执行时间
    start_time = time.time()
    image = process_a_image(input_path)
    print(f"process_a_image 耗时: {time.time() - start_time:.4f} 秒")

    aa = idx

    # 测量 sort_and_label_targets 的执行时间
    start_time = time.time()
    sorted_targets = sort_and_label_targets(results[aa].boxes.cls, results[aa].boxes.xywh)
    print(f"sort_and_label_targets 耗时: {time.time() - start_time:.4f} 秒")

    # 测量 del_txt 的执行时间
    start_time = time.time()
    processed_image = del_txt(image, results[aa].boxes.cls, results[aa].boxes.xywh)
    print(f"del_txt 耗时: {time.time() - start_time:.4f} 秒")

    op_thres = 190

    # 测量 regenerate_img 的执行时间
    start_time = time.time()
    res = regenerate_img(processed_image, op_thres)
    print(f"regenerate_img 耗时: {time.time() - start_time:.4f} 秒")

    img = 255 - res

    categories = results[aa].boxes.cls
    targets = results[aa].boxes.xywh
    gray_image = img

    image_height, image_width = gray_image.shape
    count = 1

    # 再次测量 sort_and_label_targets 的执行时间
    start_time = time.time()
    targets_sort = sort_and_label_targets(categories, targets)
    print(f"sort_and_label_targets (second call) 耗时: {time.time() - start_time:.4f} 秒")

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

        # 测量 image_to_signal_real 的执行时间
        start_time = time.time()
        signal = image_to_signal_real(target_region)
        print(f"image_to_signal_real 耗时: {time.time() - start_time:.4f} 秒")

        output_list.append(signal)

    output.append(np.array(output_list))

a = 1

