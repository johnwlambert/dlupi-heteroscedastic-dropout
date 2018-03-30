
# John Lambert

import torchvision
import pdb
import sys
sys.path.append('..')

from data_utils.localization_utils import verify_xstar_data_correct, build_synset_to_classname_dict
from cnns.data_utils.synset_utils import get_idx2synset_dict


def batch_data_sanity_check(data, step, opt):
    """
    Save to disk 8 samples of the input data (images) and privileged data (masks).
    Should be called in the _run_iteration function, as:
            batch_data_sanity_check(data, step, self.opt)
    """
    synset_to_classname_dict = build_synset_to_classname_dict()
    idx2synset_dict = get_idx2synset_dict()

    images_t, labels_t, xstar_t = data
    print('saving: ', 'step_%d_ims.png' % step)
    np_labels = labels_t.numpy()
    classnames = ''
    for i in range(8):
        synset = idx2synset_dict[np_labels[i]]
        classname = synset_to_classname_dict[synset]
        if ' ' in classname:
            classname = classname.split(' ')[0]
        if ',' in classname:
            classname = classname.split(',')[0]
        classnames += classname
    torchvision.utils.save_image(images_t[:8], 'model_inputs_sanity_check/%s_step_%d_ims_%s.png' % (opt.model_type, step, classnames),
                                 normalize=True)
    torchvision.utils.save_image(xstar_t[:8], 'model_inputs_sanity_check/%s_step_%d_masks_%s.png' % (opt.model_type, step, classnames),
                                 normalize=True)

    # verify_xstar_data_correct(images_t, image_xstar_data_t, step )
    quit()