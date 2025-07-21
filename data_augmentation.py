from data_aug_helper.XmlListConfig import *
from data_aug_helper.data_aug import *
from data_aug_helper.bbox_util import *
import os
import pickle as pkl
import numpy as np
import cv2
import pandas as pd


def xml_to_pickle(directory_name):
    path = os.getcwd() + directory_name
    file_count = 0
    for file in os.listdir(path):
        extension = file.split('.')[-1]
        filename = str(file.split('.')[0])
        new_file_loc = path + filename
        if extension == 'xml':
            file_count += 1
            file_loc = path + file
            tree = ElementTree.parse(file_loc)
            root = tree.getroot()
            xmldict = XmlDictConfig(root)
            boxes = np.empty((0, 5))
            for keys in xmldict:
                if keys == "object":
                    if str(type(xmldict[keys])) == "<class 'data_aug_helper.XmlListConfig.XmlDictConfig'>":
                        xmin = float(xmldict[keys]["bndbox"]["xmin"])
                        ymax = float(xmldict[keys]["bndbox"]["ymax"])
                        xmax = float(xmldict[keys]["bndbox"]["xmax"])
                        ymin = float(xmldict[keys]["bndbox"]["ymin"])
                        classname = xmldict[keys]["name"]
                        class_id = class_text_to_int(classname)
                        box = []
                        box.append([xmin, ymax, xmax, ymin, class_id])
                        boxes = np.append(boxes, box, axis=0)
                    else:
                        for object in xmldict[keys]:
                            xmin = float(object["bndbox"]["xmin"])
                            ymax = float(object["bndbox"]["ymax"])
                            xmax = float(object["bndbox"]["xmax"])
                            ymin = float(object["bndbox"]["ymin"])
                            classname = object["name"]
                            class_id = class_text_to_int(classname)
                            box = []
                            box.append([xmin, ymax, xmax, ymin, class_id])
                            boxes = np.append(boxes, box, axis=0)
            pkl.dump(boxes, open(new_file_loc + ".pkl", "wb"))


def read_yolo_txt(txt_file_path, image_width, image_height):
    boxes = []

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            xmin = int((x_center - width / 2) * image_width)
            ymin = int((y_center - height / 2) * image_height)
            xmax = int((x_center + width / 2) * image_width)
            ymax = int((y_center + height / 2) * image_height)

            boxes.append([xmin, ymin, xmax, ymax, class_id])

    return boxes


def create_csv_row(bnd_box, img_name, img_width, img_height):
    for row in bnd_box:
        xmin = int(row[0])
        ymin = int(row[3])
        xmax = int(row[2])
        ymax = int(row[1])

        if xmin > xmax:
            tempx = xmax
            xmax = xmin
            xmin = tempx

        if ymin > ymax:
            tempy = ymax
            ymax = ymin
            ymin = tempy

        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            break
        value = (img_name,
                 img_width,
                 img_height,
                 class_int_to_text(row[4]),
                 xmin,
                 ymin,
                 xmax,
                 ymax)
        value_list.append(value)


def save_yolo_txt(bnd_box, img_name, img_width, img_height, output_dir):
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_name)

    lines = []
    for row in bnd_box:
        xmin = float(row[0])
        ymin = float(row[3])
        xmax = float(row[2])
        ymax = float(row[1])
        class_id = int(row[4])

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            continue

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))


def horizontal_flip(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomHorizontalFlip(1)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    img_loc_name = file_path + "/" + img_name
    cv2.imwrite(img_loc_name, img_)
    save_yolo_txt(bnd_boxes_, img_name, width, height, file_path)
    index += 1
    return index


def scaling(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomScale(0.3, diff=True)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    img_loc_name = file_path + "/" + img_name
    cv2.imwrite(img_loc_name, img_)
    save_yolo_txt(bnd_boxes_, img_name, width, height, file_path)
    index += 1
    return index


def translation(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomTranslate(0.3, img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    img_loc_name = file_path + "/" + img_name
    cv2.imwrite(img_loc_name, img_)
    save_yolo_txt(bnd_boxes_, img_name, width, height, file_path)
    index += 1
    return index


def rotation(img, bnd_boxes, file_path, file_name, index):
    j = 0
    for i in range(10, 100, 10):
        j += 1
        img_, bnd_boxes_ = RandomRotate(i)(img.copy(), bnd_boxes.copy())
        img_name = file_name + str(index + j) + ".jpg"
        width = img.shape[1]
        height = img.shape[0]
        img_loc_name = file_path + "/" + img_name
        cv2.imwrite(img_loc_name, img_)
        save_yolo_txt(bnd_boxes_, img_name, width, height, file_path)
    return index + j + 1


def shearing(img, bnd_boxes, file_path, file_name, index):
    img_, bnd_boxes_ = RandomShear(0.2)(img.copy(), bnd_boxes.copy())
    img_name = file_name + str(index) + ".jpg"
    width = img.shape[1]
    height = img.shape[0]
    img_loc_name = file_path + "/" + img_name
    cv2.imwrite(img_loc_name, img_)
    save_yolo_txt(bnd_boxes_, img_name, width, height, file_path)
    index += 1
    return index


def bnd_box_data_augmentation(directory_name, Horizontal_flip=True, Scaling=True, Rotation=True, Shearing=True):
    file_count = 0
    aug_file_count = 0
    path = os.path.join(os.getcwd(), directory_name)
    new_path = os.path.join(path, 'augmented_imgs')

    if not os.path.exists(new_path):
        os.makedirs(new_path, exist_ok=True)  # Üst klasörler de yoksa yaratır

    for file in os.listdir(path):
        index = 0
        filename = file.split('.')[0]
        extension = file.split('.')[-1]
        if extension.lower() == "jpg":
            file_count += 1
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            bnd_boxes = read_yolo_txt(txt_file_path=path + filename + ".txt", image_width=img_width,
                                      image_height=img_height)
            if Horizontal_flip:
                index = horizontal_flip(img, bnd_boxes, new_path, filename, index)
            if Shearing:
                index = shearing(img, bnd_boxes, new_path, filename, index)
            if Scaling:
                index = scaling(img, bnd_boxes, new_path, filename, index)
            if Rotation:
                index = rotation(img, bnd_boxes, new_path, filename, index)
            if Shearing:
                index = shearing(img, bnd_boxes, new_path, filename, index)
            aug_file_count += index
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    values_df = pd.DataFrame(value_list, columns=column_name)
    return values_df, file_count, aug_file_count - file_count


def class_text_to_int(row_label):
    if row_label.__eq__('tassel'):
        return 1
    else:
        None


def class_int_to_text(class_id):
    if class_id.__eq__(1):
        return 'tassel'
    else:
        None


value_list = []
if __name__ == "__main__":
    print('PROCESS STARTING...')

    WORKS_NAME = 'tassel/'

    data_works_path = os.getcwd() + '/data/' + WORKS_NAME

    print('AUGMENTATION STARTING...')

    train_df, train_img_count, train_aug_img_count = bnd_box_data_augmentation(WORKS_NAME)

    train_dataset_length = len(value_list)

    value_list = []

    # print('Processing test images... ')
    # # test_csv_path = os.getcwd() + '/data/' + WORKS_NAME + '/test_labels.csv'
    # test_csv_path = os.getcwd() + '/test_labels.csv'
    # test_img_dir_name = '/training_images/' + WORKS_NAME + '/test/'
    # xml_to_pickle(test_img_dir_name)
    # test_df, test_img_count, test_aug_img_count = bnd_box_data_augmentation(test_img_dir_name)
    # test_df.to_csv(test_csv_path, index=None)
    # test_dataset_length = len(value_list)

    print("\nFINISHED...")
    print("Train img count:", train_img_count)
    print("Train augmented img count:", train_aug_img_count)
    print('Train Dataset Length: ', train_dataset_length)
    # print("Test img count:", test_img_count)
    # print("Test augmented img count:", test_aug_img_count)
    # print('Test Dataset Length: ', test_dataset_length)
