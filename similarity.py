import os
import cv2
from model import classify
from image_similarity_measures.quality_metrics import rmse, sre

img = input("File path: ")

test_img = cv2.imread(img)
data_dir = "images_compressed/" + classify(img)[:-1]

rmse_measures = {}
sre_measures = {}
scale_percent = 100  # percent of original img size
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)

for file in os.listdir(data_dir):
    try:
        img_path = os.path.join(data_dir, file)
        data_img = cv2.imread(img_path)
        resized_img = cv2.resize(data_img, dim, interpolation=cv2.INTER_AREA)
    except:
        print("Broken image found")

    rmse_measures[img_path] = rmse(test_img, resized_img)


def calc_closest_val(dict, checkMax):
    result = {}
    if (checkMax):
        closest = max(dict.values())
    else:
        closest = min(dict.values())

    for key, value in dict.items():
        if (value == closest):
            result[key] = closest
    return result


rmse = calc_closest_val(rmse_measures, False)

print("The most similar according to RMSE: ", rmse)
