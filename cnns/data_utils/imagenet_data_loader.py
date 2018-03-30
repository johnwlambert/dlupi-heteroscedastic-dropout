
# John Lambert
import os
import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_imagenet_dataset(dir, class_to_idx, num_imagenet_classes_to_use):
    """
    What does this function do?
    Args:
    -
    Returns:
    -   im_fpath_class_idx_tuples
    """
    im_fpath_class_idx_tuples = []
    targets = os.listdir(dir)
    targets.sort()
    for class_idx, target in enumerate(targets):
        if class_idx >= num_imagenet_classes_to_use:
            return im_fpath_class_idx_tuples
        print target
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    fpath_target_tuple = (path, class_to_idx[target])
                    assert (class_to_idx[target] == class_idx)
                    im_fpath_class_idx_tuples.append(fpath_target_tuple)
    return im_fpath_class_idx_tuples


def make_deterministic_imagenet_dataset(dir, class_to_idx, num_imagenet_classes_to_use, opt):
    """
    What does this function do?
    Args:
    -
    Returns:
    -   im_fpath_class_idx_tuples
    """
    im_fpath_class_idx_tuples = []
    targets = os.listdir(dir)
    targets.sort()
    for class_idx, target in enumerate(targets):
        if class_idx >= num_imagenet_classes_to_use:
            return im_fpath_class_idx_tuples
        #print target
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            fnames = sorted(fnames) # make this deterministic
            for class_example_idx, fname in enumerate(fnames):

                restriction_bool = class_example_idx >= opt.num_examples_per_class

                restriction_bool = (restriction_bool and opt.use_specific_num_examples_per_class)
                is_train_split = ('train' in dir)
                restriction_bool = (restriction_bool and is_train_split)

                if restriction_bool:
                    continue
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    fpath_target_tuple = (path, class_to_idx[target])
                    assert (class_to_idx[target] == class_idx)
                    im_fpath_class_idx_tuples.append(fpath_target_tuple)
    return im_fpath_class_idx_tuples



def find_imagenet_classes( dir):
    """
    What does this function do?
    Args:
    -   dir: string file path that points to
    Returns:
    -
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return classes, class_to_idx, idx_to_class