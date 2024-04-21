from hbac.utils.registry import * # This location becomes the new registry.

from hbac.models.eeg_model import EegModel 
from hbac.models.spectrogram_model import SpectrogramCnnModel
from hbac.models.multimodal_model import MultimodalModel, MultimodalPretrainedBackboneEntrypoint

add_registry("model", {})

register("model", "eeg_cnn_rnn_att_base", EegModel)
register("model", "spc_cnn_att_base", SpectrogramCnnModel)
register("model", "multimodal_base", MultimodalModel)
register("model", "multimodal_base_pretrained_backbone_entrypoint", MultimodalPretrainedBackboneEntrypoint)


