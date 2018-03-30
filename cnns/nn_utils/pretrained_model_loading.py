
# John Lambert

import torch
import pdb

def load_pretrained_model(model, opt):
    """ Load model weights from disk into the model that sits in main memory. """
    ckpt_dict = torch.load(opt.model_fpath)
    ckpt_state = ckpt_dict['state']
    print ('loaded ckpt with accuracy: ', ckpt_dict['acc'])
    model.load_state_dict(ckpt_state)
    return model


def load_pretrained_dlupi_model(model, opt):
    """
    Load model weights from disk into the model that sits in main memory.
    Exclude the loading of buffers.
    """
    model_fpath = opt.model_fpath
    saved_obj = torch.load(model_fpath)
    print 'Loading model with accuracy: ', saved_obj['acc']
    saved_ckpt_dict = saved_obj['state']
    curr_model_state_dict = model.state_dict()

    updated_dict = {}
    # 1. filter out unnecessary keys
    for model_key in curr_model_state_dict.keys():

        if ('running_std_1' in model_key) or ('running_std_2' in model_key):
            print( 'Skipping loading of ', model_key)
            continue
        print('     loaded weight for: ', model_key, ' from ', model_key )
        updated_dict[model_key] = saved_ckpt_dict[model_key]

    # 2. overwrite entries in the existing state dict
    curr_model_state_dict.update(updated_dict)
    # 3. load the new state dict
    model.load_state_dict(curr_model_state_dict)
    return model



def load_curriculum_learned_model(model, opt):
    """
    The weights from phase 1 and phase 2 of curriculum learning are spread out
    across 2 different ckpt files -- we load them into one common model.
    """
    saved_dlupi_obj = torch.load(opt.model_fpath) # dlupi_model_fpath
    print 'Loading DLUPI model with accuracy: ', saved_dlupi_obj['acc']
    saved_dlupi_ckpt_dict = saved_dlupi_obj['state']

    saved_curriculum_obj = torch.load(opt.curriculum_fc_weights_path)
    print 'Loading CURRICULUM Phase 2 model with accuracy: ', saved_curriculum_obj['acc']
    saved_curriculum_ckpt_dict = saved_curriculum_obj['state']

    curr_model_state_dict = model.state_dict()

    # 1. fill in from phase 1 of curriculum DLUPI model with
    updated_dict = get_renamed_pool5_vgg_weight_keys(curr_model_state_dict, saved_dlupi_ckpt_dict)

    # 2. fill in from phase 2 of curriculum
    for model_key in curr_model_state_dict.keys():
        if 'fc' in model_key:
            updated_dict[model_key] = saved_curriculum_ckpt_dict[model_key]
            print('loaded ', model_key, ' from curriculum learned keys.')

    # 3. overwrite entries in the existing state dict
    curr_model_state_dict.update(updated_dict)
    # 4. load the new state dict
    model.load_state_dict(curr_model_state_dict)
    return model


def load_pool5fn_vgg_weights( opt, pool5fn_model ):
    """
    Load the weights of a pretrained VGG network into our model, under
    the module 'x_conv_layers'. The pretrained model has the
    weights saved in a single module, 'features'.
    """
    model_fpath = opt.model_fpath

    saved_obj = torch.load(model_fpath)
    print('Loading model with accuracy: ', saved_obj['acc'] )
    saved_ckpt_dict = saved_obj['state']
    curr_model_state_dict = pool5fn_model.state_dict()

    # 1. filter out unnecessary keys
    updated_dict = get_renamed_pool5_vgg_weight_keys(curr_model_state_dict, saved_ckpt_dict)

    # 2. overwrite entries in the existing state dict
    curr_model_state_dict.update(updated_dict)
    # 3. load the new state dict
    pool5fn_model.load_state_dict(curr_model_state_dict)
    return pool5fn_model

def get_renamed_pool5_vgg_weight_keys(curr_model_state_dict, saved_ckpt_dict):
    """
    Load backbone conv weights from a traditional bernoulli dropout model
    into an x* heteroscedastic dropout model. These weights extend from conv1 to pool5.
    """
    updated_dict = {}
    for model_key in curr_model_state_dict.keys():
        saved_key = model_key.replace('x_conv_layers', 'features')
        if saved_key in saved_ckpt_dict:
            print('     loaded weight for %s as %s ' % (model_key, saved_key) )
            updated_dict[model_key] = saved_ckpt_dict[saved_key]
        else:
            print( 'key not in saved ckpt dict: ', model_key )
    return updated_dict


def load_x_fc_tower_from_bernoulli_dropout_model(model, opt):
    """
    Load model weights from disk into the model that sits in main memory.
    Model includes an X and an X_STAR tower.
    Exclude the loading of buffers.
    """
    pretrained_fc_weight_key_mapping = {
        'module.x_fc1.0.weight' : 'module.classifier.0.weight',
      'module.x_fc1.0.bias' : 'module.classifier.0.bias',
     'module.x_fc2.0.weight' : 'module.classifier.3.weight',
      'module.x_fc2.0.bias' : 'module.classifier.3.bias',
      'module.x_fc3.0.weight' : 'module.classifier.6.weight',
      'module.x_fc3.0.bias' : 'module.classifier.6.bias' }

    saved_dlupi_obj = torch.load(opt.model_fpath)
    print 'Loading FC layers from bernoulli dropout model with accuracy: ', saved_dlupi_obj['acc']
    saved_dlupi_ckpt_dict = saved_dlupi_obj['state']
    curr_model_state_dict = model.state_dict()

    updated_dict = {}
    # 1. filter out unnecessary keys
    for model_key in curr_model_state_dict.keys():

        if ('running_std_1' in model_key) or ('running_std_2' in model_key) or ('x_star' in model_key):
            continue
        if model_key in pretrained_fc_weight_key_mapping.keys():
            print('loaded ', model_key, ' from bernoulli dropout pretrained weights.')
            # fill in from phase 1 of curriculum DLUPI model with
            weight_alias = pretrained_fc_weight_key_mapping[model_key]
            updated_dict[model_key] = saved_dlupi_ckpt_dict[ weight_alias]

    # 2. overwrite entries in the existing state dict
    curr_model_state_dict.update(updated_dict)
    # 3. load the new state dict
    model.load_state_dict(curr_model_state_dict)
    return model