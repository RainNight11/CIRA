from .clip_encoder import CLIPVisionTower_VisionZip
from .llava_arch import (
    encode_images_visionzip,
    encode_images_visionzip_multi,
    prepare_inputs_labels_for_multimodal_visionzip,
    restore_image_features_sorted,
)
from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward, apply_info


def visionzip(model, dominant=191, contextual=30):
    tower = model.model.vision_tower.vision_tower
    apply_info(tower, dominant_num=int(dominant) - 1, contextual_num=int(contextual))

    from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
    from llava.model.llava_arch import LlavaMetaForCausalLM
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward
    CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
        prepare_inputs_labels_for_multimodal_visionzip
    )
    LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
    LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
    LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
    return model
