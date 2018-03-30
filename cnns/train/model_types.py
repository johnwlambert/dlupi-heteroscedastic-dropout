
# John Lambert

from enum import Enum

class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

class ModelType(AutoNumber):
    DROPOUT_FN_OF_XSTAR = ()
    DROPOUT_RANDOM_GAUSSIAN_NOISE = ()
    DROPOUT_INFORMATION = ()
    DROPOUT_BERNOULLI = ()
    MULTI_TASK_PRED_XYWH = ()
    MULTI_TASK_PRED_BW_MASK = ()
    MULTI_TASK_PRED_RGB_MASK = ()
    MODALITY_HALLUC_SHARED_PARAMS = ()
    GO_CNN_VGG = ()
    GO_CNN_RESNET = ()
    MIML_FCN_VGG = ()
    MIML_FCN_RESNET = ()
    DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE1 = ()
    DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2 = ()
    EVAL_DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2 = ()
