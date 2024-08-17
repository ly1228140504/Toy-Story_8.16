
import cv2
from sklearn.cluster import KMeans
import random
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
import os
import concurrent.futures

def get_idx(mutation_indices,grid):
    out = []
    for i in range(len(mutation_indices)):
        if np.sign(grid[mutation_indices[i][0]]) != np.sign(grid[mutation_indices[i][1]]):
            out.append(mutation_indices[i][0])
            out.append(mutation_indices[i][1])
    return out

def find_non_overlapping_pairs(mutation_indices, max_diff=5):
    # 存储最终的组合
    close_pairs = []

    # 标记已使用的数字
    used_indices = set()

    for i in range(len(mutation_indices)):
        if mutation_indices[i] in used_indices:
            continue

        for j in range(i + 1, len(mutation_indices)):
            if mutation_indices[j] in used_indices:
                continue

            # 检查差值是否在 max_diff 以内
            if abs(mutation_indices[j] - mutation_indices[i]) <= max_diff:
                close_pairs.append((mutation_indices[i], mutation_indices[j]))
                used_indices.add(mutation_indices[i])
                used_indices.add(mutation_indices[j])
                break  # 组合找到后，退出内层循环，继续找下一个数字

    return close_pairs
def find_last_non(lst):
    for i in range(len(lst) - 1, -1, -1):
        if not np.isnan(lst[i]):
            return lst[i]
    return -1  # 如果列表中所有元素都是 NaN，返回 0


def transform_edges_to_nan(signal):
    # 创建一个布尔数组，标记出非 NaN 的位置
    signal = np.array(signal)

    # 创建一个布尔数组，标记 NaN 的位置
    nan_mask = np.isnan(signal)

    # 计算左右两边是否为 NaN
    left_nan = np.roll(nan_mask, 1)  # 左边的 NaN
    right_nan = np.roll(nan_mask, -1)  # 右边的 NaN

    # 仅保留左右两边都是 NaN 的位置
    surrounded_by_nan = left_nan & right_nan & ~nan_mask

    # 将这些位置的值设为 NaN
    signal[surrounded_by_nan] = np.nan

    return signal
def grid_min(sig_past,grid,loc1,loc2):
    grid1_del = abs((loc1 - sig_past) - grid)
    grid2_del = abs((loc2 - sig_past) - grid)
    if grid1_del>grid2_del:
        return loc2 - sig_past,loc2
    else:
        return loc1 - sig_past,loc1
def grid_min_list(sig_past, grid,zero_pixels):
    grid_del = abs((zero_pixels - sig_past) - grid)
    a = np.argmin(grid_del)
    return zero_pixels[a] - sig_past,zero_pixels[a]

def find_min(num,k1,k2,the=40):
    len1 = abs(k1-num)
    len2 = abs(k2-num)
    if len1 > len2:
        if len2 > the:
            return np.nan
        else:
            return k2
    else:
        if len1 > the:
            return np.nan
        else:
            return k1
def find_the(num,k1,the=40):
    if abs(k1-num)>the:
        return 0
    else:
        return 1
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    if nyquist == 0:
        normal_cutoff=0.5
    else:
        normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    return b, a

# 应用低通滤波器
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
    # 读取灰度图像
def get_resample(resampled_signal,low_freq_signal,the = 0.016789473684210528):
    k = np.mean(low_freq_signal[int(len(low_freq_signal)*0.33):int(len(low_freq_signal)*0.66)])
    return (resampled_signal-k)*the

def image_to_signal(img_path, outpath):

    img_name_with_extension = os.path.basename(img_path)
    output_file_path = os.path.join(outpath, img_name_with_extension)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.flip(image, 0)
    # 获取图像的行和列
    rows, cols = image.shape
    the_num = np.ceil(rows*0.3)
    the_num1 = np.ceil(rows*0.5)
    # 创建一个数组来存储标记过聚类结果的图像
    clustered_image = np.zeros_like(image)

    # 为每个聚类分配不同的灰度值
    gray_values = [64, 192]  # 两个不同的灰度值

    # 处理每一列
    signal = []
    grid = []
    gird_loc = []
    gird_loc.append(0)
    grid.append(0)
    for col in range(cols):
        if col < cols*0.1:
            # 找到该列中所有的0像素点
            zero_pixels = np.array([(row, col) for row in range(rows) if image[row, col] > 210])
            # zero_pixels = zero_pixels[:,0]
            if zero_pixels.size > 2:
                # 使用K-means进行两类聚类
                zero_pixels = zero_pixels[:,0]
                zero_pixels = zero_pixels.reshape(-1, 1)

                kmeans = KMeans(n_clusters=2)


                # 选择两类
                kmeans.fit(zero_pixels)

                # 输出每列的聚类结果
                labels = kmeans.labels_
                clustered_pixels = {i: zero_pixels[labels == i] for i in range(kmeans.n_clusters)}
                loc1 = clustered_pixels[0]
                loc2 = clustered_pixels[1]
                sig_past = find_last_non(signal)
                if abs(loc1.mean() - loc2.mean()) > 3:
                    if sig_past == -1:
                        # signal.append((loc1.mean()+loc2.mean())/2)
                        signal.append((loc2.mean()))
                        # grid.append(0)
                    else:
                        grid_k,signal_k = grid_min(sig_past,grid[-1],loc1.mean(),loc2.mean())
                        signal.append(signal_k)
                        grid.append(grid_k)
                        gird_loc.append(col)
                        # signal.append(find_min(sig_past,loc1.mean(),loc2.mean(),the_num))
                else:
                    loc3 = (loc1.mean()+loc2.mean())/2
                    if sig_past == -1:
                        signal.append(loc3)
                    else:
                        if find_the(sig_past,loc3,the_num) == 1:
                            signal.append(loc3)
                            grid.append(loc3 - sig_past)
                            gird_loc.append(col)

                        else:
                            if random.random()>0.5:
                                signal.append(loc3)
                                grid.append(loc3 - sig_past)
                                gird_loc.append(col)
                            else:
                                signal.append(np.nan)
            elif zero_pixels.size == 2:
                sig_past = find_last_non(signal)
                if sig_past == -1:
                    signal.append(zero_pixels[0,0])
                else:
                    if find_the(sig_past, zero_pixels[0,0],the_num) == 1:
                        signal.append(zero_pixels[0,0])
                        grid.append(zero_pixels[0,0] - sig_past)
                        gird_loc.append(col)
                    else:
                        # if random.random() > 0.5:
                        #     signal.append(zero_pixels[0,0])
                        # else:
                        signal.append(np.nan)
            else:
                signal.append(np.nan)
        else:
            zero_pixels = np.array([(row, col) for row in range(rows) if image[row, col] > 210])
            sig_past = find_last_non(signal)
            if zero_pixels.size >= 2:
                zero_pixels = zero_pixels[:,0]
                grid_k, signal_k = grid_min_list(sig_past, grid[-1],zero_pixels)
                if find_the(sig_past, signal_k,the_num1) == 1:
                    signal.append(signal_k)
                    grid.append(grid_k)
                    gird_loc.append(col)
                else:
                    signal.append(np.nan)
            else:
                signal.append(np.nan)


    signal = np.array(signal)
    grid = np.array(grid)
    grid_loc = np.array(gird_loc)


    # 假设 arr 是一维数组
    arr =  grid

    # 计算 Z-Score
    z_scores = zscore(arr)

    # 找到 Z-Score 超过阈值的位置
    threshold = 3.5  # 自定义阈值，一般选择 2 或 3
    mutation_indices = np.where(np.abs(z_scores) > threshold)[0]


    mutation_indices =  find_non_overlapping_pairs(mutation_indices,max_diff=5)
    top_mutation_indices = get_idx(mutation_indices,grid)

    top_mutation_indices = mutation_indices

    signal[grid_loc[top_mutation_indices]] = np.nan
    signal = transform_edges_to_nan(signal)


    signal_series = pd.Series(signal)

    signal_interpolated = signal_series.interpolate(method='linear').to_numpy()
    signal_interpolated = signal_interpolated[~np.isnan(signal_interpolated)]





    fs = 100*len(signal_interpolated)/250 # 采样频率


    cutoff = 1# 截止频率（Hz）
    order = 5
    # 提取低频成分
    low_freq_signal = lowpass_filter(signal_interpolated, cutoff, fs, order)
    # plt.figure(figsize=(40, 8))
    # plt.plot(signal_interpolated,linewidth=8)
    # plt.plot(low_freq_signal,linewidth=8,color='orange')
    # plt.savefig(output_file_path, bbox_inches='tight')
    # plt.close()  # 关闭图像以避免显示

    target_length = 250
    original_indices = np.linspace(0, len(signal_interpolated) - 1, num=len(signal_interpolated))
    target_indices = np.linspace(0, len(signal_interpolated) - 1, num=target_length)
    # 使用 np.interp 进行线性插值来调整长度
    resampled_signal = np.interp(target_indices, original_indices, signal_interpolated)
    output = get_resample(resampled_signal,low_freq_signal)
    return output
    # plt.figure(figsize=(40, 8))
    # plt.plot(resampled_signal, linewidth=16)
    # plt.show()



def process_images_in_folder(folder_path, outpath):
    # 获取文件夹中所有图片的路径
    img_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 file.endswith(('.jpg', '.png', '.jpeg'))]

    # 创建输出文件夹（如果不存在）
    os.makedirs(outpath, exist_ok=True)

    # 使用 ThreadPoolExecutor 进行并行处理
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交所有图片的处理任务
        futures = {executor.submit(image_to_signal, img_path, outpath): img_path for img_path in img_paths}

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            img_path = futures[future]
            try:
                result = future.result()
                print(f"Completed processing {img_path} -> {result}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def process_images_in_folder1(folder_path, outpath):
    # 获取文件夹中所有图片的路径
    img_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 file.endswith(('.jpg', '.png', '.jpeg'))]

    # 创建输出文件夹（如果不存在）
    os.makedirs(outpath, exist_ok=True)
    for i in range(len(img_paths)):
        image_to_signal(img_paths[i], outpath)
        print(img_paths[i])

# folder_path = 'E:/code_crazy/pnet2024-1/pnet2024/yolo/image3_3/'
# outpath = 'E:/code_crazy/pnet2024-1/pnet2024/yolo/image3_4/'
# # image_to_signal(folder_path, outpath)
# process_images_in_folder1(folder_path, outpath)






def image_to_signal_real(img,target_nums):

    # img_name_with_extension = os.path.basename(img_path)
    # output_file_path = os.path.join(outpath, img_name_with_extension)
    # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = img
    image = cv2.flip(image, 0)
    # 获取图像的行和列
    rows, cols = image.shape
    the_num = np.ceil(rows*0.3)
    the_num1 = np.ceil(rows*0.8)
    # 创建一个数组来存储标记过聚类结果的图像
    clustered_image = np.zeros_like(image)


    # 处理每一列
    signal = []
    grid = []
    gird_loc = []
    gird_loc.append(0)
    grid.append(0)
    for col in range(cols):
        if col < cols*0.1:
            # 找到该列中所有的0像素点
            zero_pixels = np.array([(row, col) for row in range(rows) if image[row, col] > 210])
            # zero_pixels = zero_pixels[:,0]
            if zero_pixels.size > 2:
                # 使用K-means进行两类聚类
                zero_pixels = zero_pixels[:,0]
                zero_pixels = zero_pixels.reshape(-1, 1)

                kmeans = KMeans(n_clusters=2)


                # 选择两类
                kmeans.fit(zero_pixels)

                # 输出每列的聚类结果
                labels = kmeans.labels_
                clustered_pixels = {i: zero_pixels[labels == i] for i in range(kmeans.n_clusters)}
                loc1 = clustered_pixels[0]
                loc2 = clustered_pixels[1]
                sig_past = find_last_non(signal)
                if abs(loc1.mean() - loc2.mean()) > 3:
                    if sig_past == -1:
                        # signal.append((loc1.mean()+loc2.mean())/2)
                        signal.append((loc2.mean()))
                        # grid.append(0)
                    else:
                        grid_k,signal_k = grid_min(sig_past,grid[-1],loc1.mean(),loc2.mean())
                        signal.append(signal_k)
                        grid.append(grid_k)
                        gird_loc.append(col)
                        # signal.append(find_min(sig_past,loc1.mean(),loc2.mean(),the_num))
                else:
                    loc3 = (loc1.mean()+loc2.mean())/2
                    if sig_past == -1:
                        signal.append(loc3)
                    else:
                        if find_the(sig_past,loc3,the_num) == 1:
                            signal.append(loc3)
                            grid.append(loc3 - sig_past)
                            gird_loc.append(col)

                        else:
                            if random.random()>0.5:
                                signal.append(loc3)
                                grid.append(loc3 - sig_past)
                                gird_loc.append(col)
                            else:
                                signal.append(np.nan)
            elif zero_pixels.size == 2:
                sig_past = find_last_non(signal)
                if sig_past == -1:
                    signal.append(zero_pixels[0,0])
                else:
                    if find_the(sig_past, zero_pixels[0,0],the_num) == 1:
                        signal.append(zero_pixels[0,0])
                        grid.append(zero_pixels[0,0] - sig_past)
                        gird_loc.append(col)
                    else:
                        # if random.random() > 0.5:
                        #     signal.append(zero_pixels[0,0])
                        # else:
                        signal.append(np.nan)
            else:
                signal.append(np.nan)
        else:
            zero_pixels = np.array([(row, col) for row in range(rows) if image[row, col] > 210])
            sig_past = find_last_non(signal)
            if zero_pixels.size >= 2:
                zero_pixels = zero_pixels[:,0]
                grid_k, signal_k = grid_min_list(sig_past, grid[-1],zero_pixels)
                if find_the(sig_past, signal_k,the_num1) == 1:
                    signal.append(signal_k)
                    grid.append(grid_k)
                    gird_loc.append(col)
                else:
                    signal.append(np.nan)
            else:
                signal.append(np.nan)


    signal = np.array(signal)
    grid = np.array(grid)
    grid_loc = np.array(gird_loc)


    # 假设 arr 是一维数组
    arr =  grid

    # 计算 Z-Score
    z_scores = zscore(arr)

    # 找到 Z-Score 超过阈值的位置
    threshold = 3.5  # 自定义阈值，一般选择 2 或 3
    mutation_indices = np.where(np.abs(z_scores) > threshold)[0]


    mutation_indices =  find_non_overlapping_pairs(mutation_indices,max_diff=5)
    top_mutation_indices = get_idx(mutation_indices,grid)

    top_mutation_indices = top_mutation_indices
    print(top_mutation_indices)
    signal = signal.astype(float)
    # signal[grid_loc[top_mutation_indices]] = 0
    signal[grid_loc[top_mutation_indices]] = np.nan
    signal = transform_edges_to_nan(signal)


    signal_series = pd.Series(signal)

    signal_interpolated = signal_series.interpolate(method='linear').to_numpy()
    signal_interpolated = signal_interpolated[~np.isnan(signal_interpolated)]




    fs = 100*len(signal_interpolated)/250 # 采样频率
    print(f"fs:{fs}")
    if fs < 20 or len(signal_interpolated)<18:
        return np.zeros(target_nums)
    else:
        cutoff = 1# 截止频率（Hz）
        order = 5
        # 提取低频成分
        low_freq_signal = lowpass_filter(signal_interpolated, cutoff, fs, order)
        # plt.figure(figsize=(40, 8))
        # plt.plot(signal_interpolated,linewidth=8)
        # plt.plot(low_freq_signal,linewidth=8,color='orange')
        # plt.savefig(output_file_path, bbox_inches='tight')
        # plt.close()  # 关闭图像以避免显示

        target_length = target_nums
        original_indices = np.linspace(0, len(signal_interpolated) - 1, num=len(signal_interpolated))
        target_indices = np.linspace(0, len(signal_interpolated) - 1, num=target_length)
        # 使用 np.interp 进行线性插值来调整长度
        resampled_signal = np.interp(target_indices, original_indices, signal_interpolated)
        output = get_resample(resampled_signal,low_freq_signal)
        return output
