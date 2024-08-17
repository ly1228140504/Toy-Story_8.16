import cv2
import numpy as np
import matplotlib.pyplot as plt
# import lowrank_cons
import concurrent.futures
import os

# Hyperparameter
k = 3  # 放大倍数


# image = cv2.imread('D:/AI/CinC2024/ptb-xl/op/100-00000/00007_lr-0.png', cv2.IMREAD_GRAYSCALE)
# # image = cv2.bitwise_not(image)
# U, s, Vt = np.linalg.svd(image)
# print(s[0],s[-1])
# s = s * k
# print(s[0]/s[-1])

# # U = U[:, 0:s.shape[0]]
# # Vt = Vt[0:s.shape[0], :]

# S = np.diag(s)
# zerosmatrix = np.zeros((s.shape[0], Vt.shape[0]-s.shape[0]))
# S = np.hstack((S,zerosmatrix))
# F = np.dot(np.dot(U,S), Vt)

# # cv2.imwrite('D:/AI/CinC2024/ptb-xl/op_denoise/100-00000/0000'+str(i)+'_lr_dn-0.png', F)
# cv2.imwrite('D:/AI/CinC2024/00001_lr_dn-0.png', F)

def process_a_image(image_path,k=3):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1700, 1500))

    U, s, Vt = np.linalg.svd(image)
    # print(f"Image: {os.path.basename(image_path)} - s[0]: {s[0]}, s[-1]: {s[-1]}")
    s = s * k
    # print(f"Image: {os.path.basename(image_path)} - s[0]/s[-1]: {s[0] / s[-1]}")

    S = np.diag(s)
    zerosmatrix = np.zeros((s.shape[0], Vt.shape[0] - s.shape[0]))
    S = np.hstack((S, zerosmatrix))
    F = np.dot(np.dot(U, S), Vt)

    # output_path = os.path.join(output_folder, os.path.basename(image_path))
    # cv2.imwrite(output_path, F)
    # F1 = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    # F = np.ceil((F - np.min(F)) / (np.max(F) - np.min(F)) * 255)

    return F

# path = r'D:\pycharm_software\PyCharm 2024.1\codes\race\compe\python-example-2024-main\python-example-2024-main\00000\00019_hr-0.png'
# a = process_a_image(path,k=3)
# cv2.imwrite('random_image_gray.png', a)
# a = 1
def process_image(image_path, output_folder, k):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    U, s, Vt = np.linalg.svd(image)
    print(f"Image: {os.path.basename(image_path)} - s[0]: {s[0]}, s[-1]: {s[-1]}")
    s = s * k
    print(f"Image: {os.path.basename(image_path)} - s[0]/s[-1]: {s[0] / s[-1]}")

    S = np.diag(s)
    zerosmatrix = np.zeros((s.shape[0], Vt.shape[0] - s.shape[0]))
    S = np.hstack((S, zerosmatrix))
    F = np.dot(np.dot(U, S), Vt)

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, F)
    F1 = cv2.imread(output_path,cv2.IMREAD_GRAYSCALE)
    # F = np.ceil((F - np.min(F)) / (np.max(F) - np.min(F)) * 255)
    F = F//3
    print(f"Saved processed image to {output_path}")


def process_images_in_folder(input_folder, output_folder, k):
    print(1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]
    print(1)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, image_path, output_folder, k)
            for image_path in image_paths
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")


def process_images(input_folder, output_folder, k):
    print(1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]
    for image_path in image_paths:
        process_image(image_path, output_folder, k)


# # input_folder = r'D:\pnet2024\yolo\image'
# input_folder = r'F:\op\op_roat\datasets_1\images\test'
# output_folder = r'D:\pnet2024\yolo\image_spect'
# process_images(input_folder, output_folder, k)

