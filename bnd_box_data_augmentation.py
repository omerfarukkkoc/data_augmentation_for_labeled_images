from data_aug_helper.data_aug import *
from data_aug_helper.bbox_util import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
from PIL import Image
import pandas as pd

value_list = []


def class_int_to_text(class_id):
    if class_id.__eq__(1):
        return 'tassel'
    else:
        None


def horizontal_flip(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomHorizontalFlip(1)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    for row in bnd_boxes_:
        value = (img_name,
                 width,
                 height,
                 class_int_to_text(row[4]),
                 int(row[0]),
                 int(row[3]),
                 int(row[2]),
                 int(row[1]))
        value_list.append(value)

    img_loc_name = file_path + img_name
    print(img_loc_name, ' saving')
    cv2.imwrite(img_loc_name, img_)
    # pkl.dump(bnd_boxes_, open(file_path + file_name + str(index) + ".pkl", "wb"))
    index += 1
    return index


def scaling(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomScale(0.3, diff=True)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    for row in bnd_boxes_:
        value = (img_name,
                 width,
                 height,
                 class_int_to_text(row[4]),
                 int(row[0]),
                 int(row[3]),
                 int(row[2]),
                 int(row[1]))
        value_list.append(value)

    img_loc_name = file_path + img_name
    print(img_loc_name, ' saving')
    cv2.imwrite(img_loc_name, img_)
    # pkl.dump(bnd_boxes_, open(file_path + file_name + str(index) + ".pkl", "wb"))
    index += 1
    return index


def translation(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomTranslate(0.3, img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    for row in bnd_boxes_:
        value = (img_name,
                 width,
                 height,
                 class_int_to_text(row[4]),
                 int(row[0]),
                 int(row[3]),
                 int(row[2]),
                 int(row[1]))
        value_list.append(value)

    img_loc_name = file_path + img_name
    print(img_loc_name, ' saving')
    cv2.imwrite(img_loc_name, img_)
    # pkl.dump(bnd_boxes_, open(file_path + file_name + str(index) + ".pkl", "wb"))
    index += 1
    return index

def rotation(img, bnd_boxes, file_path, file_name, index):
    j = 0
    for i in range(10, 360, 10):
        j += 1
        img_, bnd_boxes_ = RandomRotate(i)(img.copy(), bnd_boxes.copy())
        img_name = file_name + str(index + j) + ".jpg"
        width = img.shape[1]
        height = img.shape[0]
        for row in bnd_boxes_:
            value = (img_name,
                     width,
                     height,
                     class_int_to_text(row[4]),
                     int(row[0]),
                     int(row[3]),
                     int(row[2]),
                     int(row[1]))
            value_list.append(value)

        img_loc_name = file_path + img_name
        print(img_loc_name, ' saving')
        cv2.imwrite(img_loc_name, img_)
        # pkl.dump(bnd_boxes_, open(file_path + file_name + str(index + j) + ".pkl", "wb"))
    return index + j + 1


def shearing(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomShear(0.2)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    for row in bnd_boxes_:
        value = (img_name,
                 width,
                 height,
                 class_int_to_text(row[4]),
                 int(row[0]),
                 int(row[3]),
                 int(row[2]),
                 int(row[1]))
        value_list.append(value)

    img_loc_name = file_path + img_name
    print(img_loc_name, ' saving')
    cv2.imwrite(img_loc_name, img_)
    # pkl.dump(bnd_boxes_, open(file_path + file_name + str(index) + ".pkl", "wb"))
    index += 1
    return index


def main(directory_name):
    file_count = 0
    aug_file_count = 0
    path = os.getcwd() + directory_name
    new_path = os.getcwd() + directory_name + 'augmented_imgs/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for file in os.listdir(path):
        index = 0
        filename = file.split('.')[0]
        extension = file.split('.')[-1]
        if extension == "jpg":
            file_count += 1
            img = cv2.imread(path + file)
            bnd_boxes = pkl.load(open(path + filename + ".pkl", "rb"))
            index = horizontal_flip(img, bnd_boxes, new_path, filename, index)
            index = scaling(img, bnd_boxes, new_path, filename, index)
            index = rotation(img, bnd_boxes, new_path, filename, index)
            index = shearing(img, bnd_boxes, new_path, filename, index)
            aug_file_count += index

    print("\nFinished...\nFile Count:", file_count)
    print("Augmented File Count: ", aug_file_count - file_count)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    values_df = pd.DataFrame(value_list, columns=column_name)
    values_df.to_csv('aaa.csv', index=None)


if __name__ == "__main__":
    dir_name = '/imgs/'
    main(dir_name)
