
# John Lambert

from cnns.base_networks.dcgan_multitask import DCGAN_Multitask_Autoencoder
from cnns.base_networks.deconvnet import DeconvModuleTo224
from cnns.base_networks.vgg_multitask import VGG_pred_xywh
from cnns.models.vgg_dlupi_model import DualNetworksVGG
from cnns.base_networks.vgg import vgg16_bn
from cnns.models.information_dropout import VGG_InformationDropout
from cnns.base_networks.random_gauss_dropout_vgg import vgg16_bn_random_gaussian_dropout
from cnns.models.shared_params_modality_hallucination_model import SharedParamsModalityHallucinationModel
from cnns.models.miml_vgg_model import MIML_VGG
from cnns.models.miml_resnet_model import miml_resnet50
from cnns.models.group_orthogonal_cnn_model import GoCNNModel

from cnns.train.model_types import ModelType

def build_model(opt):
    """ Return the model on the GPU """
    print opt.model_type
    if (opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR) or \
        (opt.model_type == ModelType.EVAL_DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2):
        print('We are using DualNetworks VGG for DLUPI')
        return DualNetworksVGG(opt).cuda()

    if opt.model_type == ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE:
        # noise in train, but not in test...
        return vgg16_bn_random_gaussian_dropout().cuda()

    if opt.model_type == ModelType.DROPOUT_INFORMATION:
        return VGG_InformationDropout(opt).cuda()

    if opt.model_type == ModelType.DROPOUT_BERNOULLI:
        print('Build VGG NO XSTAR with ', opt.num_classes, ' classes.')
        return vgg16_bn(num_classes=opt.num_classes,
                        image_size=opt.image_size).cuda()

    if opt.model_type == ModelType.MULTI_TASK_PRED_XYWH:
        return VGG_pred_xywh(opt).cuda()  # VGG-16 that predicts x,y,w,h

    if opt.model_type == ModelType.MULTI_TASK_PRED_BW_MASK:
        return DeconvModuleTo224(use_bw_mask=True, opt=opt).cuda()
        # return DCGAN_Multitask_Autoencoder(self.opt).cuda()

    if opt.model_type == ModelType.MULTI_TASK_PRED_RGB_MASK:
        return DeconvModuleTo224(use_bw_mask=False, opt=opt).cuda()

    if opt.model_type == ModelType.MODALITY_HALLUC_SHARED_PARAMS:
        return SharedParamsModalityHallucinationModel(opt).cuda()

    if opt.model_type == ModelType.MIML_FCN_RESNET:
        print('Building MIML ResNet-50 model...')
        return miml_resnet50(opt).cuda()  # DualNetworks_MIML_FCN(opt).cuda()

    if opt.model_type == ModelType.MIML_FCN_VGG:
        print('Building MIML VGG-16 model...')
        return MIML_VGG(opt).cuda()

    if opt.model_type == ModelType.GO_CNN_VGG:
        print('Building GoCNN VGG model...')
        return GoCNNModel(use_vgg=True, opt=opt).cuda()

    if opt.model_type == ModelType.GO_CNN_RESNET:
        print('Building GoCNN ResNet model...')
        return GoCNNModel(use_vgg=False, opt=opt).cuda()

    else:
        print 'undefined model type. quitting...'
        quit()