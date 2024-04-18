import os
import shutil
from loguru import logger

DATASET_PATH = "./dataset/"
DATA_COUNT = 0
TOTAL_COUNT = 5

def main():
    dirs = os.listdir(DATASET_PATH)
    logger.debug(dirs)
    for dir in dirs:
        if os.path.isdir(DATASET_PATH + dir):
            # choose the class_id = head 
            if dir == "detection2":
                useClassId = 1
            elif dir == "tracker":
                useClassId = 1
            else:
                useClassId = 0

            files = os.listdir(DATASET_PATH + dir)
            assert 'train' in files and 'valid' in files, "train and valid must be in the dataset"
            train_dir = os.listdir(os.path.join(DATASET_PATH, dir, 'train'))
            val_dir = os.listdir(os.path.join(DATASET_PATH, dir, 'valid'))
            assert 'images' in train_dir and 'labels' in train_dir, "images and labels must be in the train directory"
            assert 'images' in val_dir and 'labels' in val_dir, "images and labels must be in the valid directory"

            train_images_path = os.path.join(DATASET_PATH, dir, 'train', 'images')
            train_labels_path = os.path.join(DATASET_PATH, dir, 'train', 'labels')
            val_images_path = os.path.join(DATASET_PATH, dir, 'valid', 'images')
            val_labels_path = os.path.join(DATASET_PATH, dir, 'valid', 'labels')
            
            tansMerge(images_path=train_images_path, labels_path=train_labels_path, type='train', valid_id=useClassId)
            tansMerge(images_path=val_images_path, labels_path=val_labels_path, type='valid', valid_id=useClassId)

def tansMerge(images_path, labels_path, type, valid_id):
    global DATA_COUNT
    if type == 'train':
        save_path = os.path.abspath(os.path.join(DATASET_PATH, '../merge_dataset','train'))
    else:
        save_path = os.path.abspath(os.path.join(DATASET_PATH, '../merge_dataset','val'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    images = os.listdir(images_path)
    labels = os.listdir(labels_path)
    # logger.info(f"Dataset: {dir} has {len(images)} images for training")
    # logger.info(f"Dataset: {dir} has {len(val_images)} images for val")
    for image in images:
        label_name = image.replace('.jpg', '.txt')
        if label_name not in labels:
            logger.warning("Could not find label for image, ignoring")
        with open(os.path.join(labels_path, label_name), 'r') as original_f:
            lines = original_f.readlines()
            re_lines = ""
            for line in lines:
                line = line.strip().split(' ')
                if line[0] == str(valid_id):
                    re_lines = re_lines + '0' + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] + ' ' + line[4] + '\n'
            # print(re_lines)
            prefix = "{:0>5}".format(DATA_COUNT)
            DATA_COUNT += 1
            save_label_path = os.path.join(save_path, prefix + '.txt')
            save_image_path = os.path.join(save_path, prefix + '.jpg')
            with open(save_label_path, 'a') as new_f:
                new_f.write(re_lines)
            shutil.copyfile(os.path.join(images_path, image), save_image_path)


if __name__ == '__main__':
    main()