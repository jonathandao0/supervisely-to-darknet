"""Create a the Supervisely dataset format into Darknet-COCO format.

The standard Supervisely dataset format contains the images (*.jpg) 
and the labels (*.json). This file converts the dataset as a folder 
into another folder in Darket-COCO format.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
from tqdm import tqdm


# Get names of classes from meta.json and write them into *.names file
# In reference to https://github.com/ultralytics/JSON2YOLO/blob/177e96ad79bb1832c82dc3a1cec6681329ee1835/run.py#L36
def get_classes(read_name, write_name=None, write=True):
    # Import JSON
    with open(read_name) as f:
        data = json.load(f)

    # Get classes from "classes" - "title" values
    classes_object = data['classes']
    classes = []
    for class_object in classes_object:
        class_name = class_object['title']
        classes.append(class_name)
        
        # Write *.names file
        if write is True:
            with open(write_name + 'classes.names', 'a') as nf:
                nf.write('{}\n'.format(class_name))
    return classes, './classes.names'
  
# Create folders: images and labels
# https://github.com/ultralytics/JSON2YOLO/blob/177e96ad79bb1832c82dc3a1cec6681329ee1835/utils.py#L73
def make_folders(path='./dataset/'):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    os.makedirs(path + os.sep + 'images')  # make new labels folder

# Random split: get random indices
def split_indices(data, train_ratio=0.9, val_ratio=0.1, shuffle=True):
    test_ratio = 1 - train_ratio - val_ratio
    indices = np.arange(len(data))
    if shuffle == True:
        np.random.shuffle(indices)
    end_train = round(len(data) * train_ratio)
    end_val = round(len(data) * val_ratio + end_train)
    end_test = round(len(data) * test_ratio + end_val)

    return indices[:end_train], indices[end_train:end_val], indices[end_val:end_test]

# Random split: split the paths
def split_paths(new_data_name, img_paths, split_shuffle, train_size, val_size):
    out_path = './' + new_data_name + os.sep
    train_ids, val_ids, test_ids = split_indices(img_paths, train_size, val_size, split_shuffle)
    datasets = {'train': train_ids, 'validation': val_ids, 'test': test_ids}
    sets_paths = {} # Store the paths of subsets *.txt files
    for key, ids in datasets.items():
        if ids.any():
            sets_paths[key] = './' + new_data_name + '_' + key + '.txt' # Store the key/path
            with open(out_path + new_data_name + '_' + key + '.txt', 'a') as wf:
                for idx in tqdm(ids, desc=key + ' paths'):
                    wf.write('{}'.format(img_paths[idx]) + '\n')
    return sets_paths


# Convert from Supervisely format to darknet format.
# In reference to https://github.com/ultralytics/JSON2YOLO/blob/177e96ad79bb1832c82dc3a1cec6681329ee1835/run.py#L10
def convert_supervisely_json(read_path, new_data_name, meta_file, split_shuffle, train_size, val_size):
    # Create folders
    out_path = './' + new_data_name + os.sep
    make_folders(out_path)

    # Write classes.names from meta.json
    classes, names_path = get_classes(meta_file, out_path)

    # Get all file real paths
    read_path = read_path + os.sep
    ann_paths = sorted(glob.glob(read_path + 'ann/' + '*.json'))
    img_paths = sorted(glob.glob(read_path + 'img/' + '*.[jp][np]g'))

    # Import all json annotation files for images
    for (ann_path, img_path) in tqdm(zip(ann_paths, img_paths), desc='Annotations'):
        label_name = os.path.basename(img_path)[:-4] + '.txt'

        # Import json
        with open(ann_path) as ann_f:
            ann_data = json.load(ann_f)
        
        # Image size
        image_size = ann_data['size']   # dict: {'height': , 'width': }

        # Objects bounding boxes
        bboxes = ann_data['objects']
        if len(bboxes) != 0:    # With object(s)
            for bbox in bboxes:
                class_index = classes.index(bbox['classTitle'])
                corner_coords = bbox['points']['exterior']  # bbox corner coordinates in [[left, top], [right, bottom]]

                # Normalisation
                b_x_center = (corner_coords[0][0] + corner_coords[1][0]) / 2 / image_size['width']
                b_y_center = (corner_coords[0][1] + corner_coords[1][1]) / 2 / image_size['height']
                b_width = (corner_coords[1][0] - corner_coords[0][0]) / image_size['width']
                b_height = (corner_coords[1][1] - corner_coords[0][1]) / image_size['height']

                b_width = 0 if (b_width < 0.) else b_width
                b_height = 0 if (b_height < 0.) else b_height

                # Write labels file
                if (b_width >= 0.) and (b_height >= 0.):
                    with open(out_path + 'labels/' + label_name, 'a') as label_f:
                        label_f.write('%d %.6f %.6f %.6f %.6f\n' % (class_index, b_x_center, b_y_center, b_width, b_height))
        
            # Move images to images folder
            shutil.copy(img_path, out_path + 'images/')
    
    # Split training set
    img_paths = sorted(glob.glob(out_path + 'images/' + '*.[jp][np]g'))
    sets_paths = split_paths(new_data_name, img_paths, split_shuffle, train_size, val_size)

    # Write .data file
    with open(out_path + new_data_name + '.data', 'a') as data_f:
        data_f.write('classes={}\n'.format(len(classes)))   # Classes count
        data_f.write('train={}\n'.format(sets_paths['train']))   # Path of train set
        data_f.write('valid={}\n'.format(sets_paths['validation'])) # Path of validation set
        data_f.write('names={}\n'.format(names_path))   # Path of .names file


    # Summary
    print('Done. Dataset saved to %s' % (os.getcwd() + os.sep + new_data_name + os.sep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--origin', type=str, default='P30__P30_04', help='The name of original data downloaded from Supervisely.')
    parser.add_argument('-d', '--dest', type=str, default='P30', help='The name of the output dataset folder.')
    parser.add_argument('-m', '--meta', type=str, default='meta.json', help='The name of the meta file of the data.')
    parser.add_argument('-s', '--shuffle', action='store_true', help='Whether to randomly split image set.')
    parser.add_argument('-t', '--train-size', type=float, default=0.9, help='Percentage of train set.')
    parser.add_argument('-v', '--val-size', type=float, default=0.1, help='Percentage of validation set.')
    opt = parser.parse_args()
    print(opt)

    convert_supervisely_json(
        read_path = opt.origin,
        new_data_name = opt.dest,
        meta_file = opt.meta,
        split_shuffle = opt.shuffle,
        train_size = opt.train_size,
        val_size = opt.val_size
    )
