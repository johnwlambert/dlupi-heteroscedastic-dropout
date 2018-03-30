
# John Lambert

import torch
from cnns.train.model_types import ModelType

def feedforward_routine(model, images_v, image_xstar_data_t, train, opt):
    if opt.model_type == ModelType.MODALITY_HALLUC_SHARED_PARAMS:
        x_output_v =feedforward_modality_halluc(model, images_v, image_xstar_data_t, train)

    elif opt.model_type in [ ModelType.MULTI_TASK_PRED_XYWH, ModelType.MULTI_TASK_PRED_BW_MASK,
        ModelType.MULTI_TASK_PRED_RGB_MASK ]:
        x_output_v = feedforward_multi_task(model, images_v )

    elif opt.model_type in [ModelType.GO_CNN_VGG, ModelType.GO_CNN_RESNET]:
        x_output_v = model(images_v, None, None, train)

    elif opt.model_type == ModelType.DROPOUT_INFORMATION:
        x_output_v, _ = model(images_v, train)

    elif opt.model_type == ModelType.DROPOUT_BERNOULLI:
        x_output_v = model(images_v)

    elif opt.model_type == ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE:
        x_output_v = model(images_v, train)

    else:
        # DROPOUT_FN_OF_XSTAR
        # MIML_FCN_VGG
        # MIML_FCN_RESNET
        x_output_v, _ = model( images_v, image_xstar_data_t, train)

    return x_output_v


def feedforward_modality_halluc(model, images_v, xstar_v, train ):

    logits_cache = model(images_v, xstar_v, train)
    halluc_logits, rgb_logits = logits_cache

    # this is all we need for test time
    # take preds before and after softmax, assert that equal, because exp preserves order...
    concat_rgb_halluc_logits = torch.cat([rgb_logits.unsqueeze(0), halluc_logits.unsqueeze(0)], 0)
    rgb_halluc_avg_logits = torch.mean(concat_rgb_halluc_logits, 0)
    rgb_halluc_avg_logits = rgb_halluc_avg_logits.squeeze()

    return rgb_halluc_avg_logits


def feedforward_multi_task(model, images_v):
    _, class_logits = model(images_v)
    return class_logits