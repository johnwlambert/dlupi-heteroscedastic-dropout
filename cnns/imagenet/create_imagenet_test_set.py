
# John Lambert

import os
import numpy as np
import cPickle as pkl
from shutil import copyfile
import pdb

# create a test set for ImageNet

# full dataset has:  1,331,167
# loc. dataset has:  594,546

NUM_CLASSES = 1000
NUM_VAL_EXAMPLES_PER_CLASS = 50
PRINT_EVERY =50

FULL_DATASET_PATH = '/vision/group/ImageNet_2012'
LOCALIZATION_DATASET_PATH = '/vision/group/ImageNetLocalization'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def gather_split_tuples(dir):
    print 'searching ', dir, '...'
    split_fpaths = []
    targets = os.listdir(dir)
    targets.sort()
    for class_idx, target in enumerate(targets):
        if class_idx % PRINT_EVERY == 0:
            print 'on class #', class_idx, ', we have ', len(split_fpaths), ' examples so far'
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            # Render the directory traversal deterministic
            fnames = sorted(fnames)
            for class_example_idx, fname in enumerate(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    split_fpaths.append(path)

    return split_fpaths


def pkl_load_dict(dict_fpath):
    """ Read python dictionary back into memory from a file. """
    f = open(dict_fpath, 'rb')
    dict = pkl.load(f)
    f.close()
    return dict


def pkl_save_dict(dict_fpath, dict):
    """ Save in-memory python dictionary to disk. """
    f = open(dict_fpath, 'wb')
    pkl.dump(dict, f)
    f.close()


def count_files():
    """
    Gather all the examples i'm using across 3 splits of ImageNetLocalization folder
    Then count up the ImageNet 2012 folder
    Identify 5,000-50,000 unused examples
    """
    create_dict_of_dir_children_files(LOCALIZATION_DATASET_PATH, ['train', 'val'])
    create_dict_of_dir_children_files(FULL_DATASET_PATH, ['train', 'val'])


def create_dict_of_dir_children_files(dir_path, subdirs):
    """
    Iterate through a dataset directory and save the paths to its children
    (stored in a python dictionary) to disk via pkl. Count the number of files found.
    """
    fpaths = []
    dataset_dict = {}
    for subdir_name in subdirs:
        subdir_paths = gather_split_tuples( os.path.join( dir_path, subdir_name))
        for k,v in enumerate(subdir_paths):
            dataset_dict[v] = subdir_name
        fpaths.extend(subdir_paths)

    print '%s dataset has %d files' % (dir_path, len(fpaths))
    pkl_save_dict( dir_path + '_files.pkl', dataset_dict)




def create_holdout_test_set():
    LOC_TEST_DIR = os.path.join( LOCALIZATION_DATASET_PATH, 'test' )

    if not os.path.isdir(LOC_TEST_DIR):
        os.makedirs(LOC_TEST_DIR)

    full_dataset_dict = pkl_load_dict( os.path.join(LOCALIZATION_DATASET_PATH, 'ImageNet_2012_files.pkl'  ))
    loc_dataset_dict = pkl_load_dict( os.path.join(LOCALIZATION_DATASET_PATH, 'ImageNetLocalization_files.pkl' ))

    synset_to_idx = {}
    synset_counts = np.zeros(NUM_CLASSES)

    for fpath, corr_split in full_dataset_dict.iteritems():
        synset = fpath.split('/')[-2]

        SYNSET_LOC_TEST_DIR = os.path.join( LOC_TEST_DIR, synset )

        if not os.path.isdir(SYNSET_LOC_TEST_DIR):
            os.makedirs(SYNSET_LOC_TEST_DIR)

        if synset not in synset_to_idx:
            synset_idx = len(synset_to_idx.keys() )
            synset_to_idx[synset] = synset_idx
        else:
            synset_idx = synset_to_idx[synset]

        if synset_counts[synset_idx] >= NUM_VAL_EXAMPLES_PER_CLASS:
            continue

        fname =fpath.split('/')[-1]
        potential_loc_train_fpath = os.path.join( LOCALIZATION_DATASET_PATH, 'train', synset, fname)
        potential_loc_val_fpath =os.path.join( LOCALIZATION_DATASET_PATH, 'val', synset, fname)

        if potential_loc_train_fpath in loc_dataset_dict:
            continue
        if potential_loc_val_fpath in loc_dataset_dict:
            continue
        src = fpath
        dst = os.path.join( LOCALIZATION_DATASET_PATH, 'test', synset, fname)
        #print 'we would copy from src: ', src, ' to dst: ', dst
        copyfile(src, dst)
        synset_counts[synset_idx] += 1

        if np.sum( synset_counts) % 100 == 0:
            print synset_counts


if __name__ == '__main__':
    create_holdout_test_set()

    #     loc_dataset_dict = {k: v for v, k in enumerate(loc_dataset_fpaths)}
    # full_dataset_dict = {k: v for v, k in enumerate(full_dataset_fpaths)}
    # full_dataset_dict = {k: v for v, k in enumerate(full_dataset_fpaths)}