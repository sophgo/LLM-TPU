from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers import Qwen3Config


class StepRoboticsVisionEncoderConfig(PretrainedConfig):

    def __init__(
        self,
        width=1536,
        layers=47,
        heads=16,
        num_channels=3,
        image_size=728,
        mlp_ratio = 8960/1536,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        ues_cls_token=False,
        use_ln_pre=True,
        use_ln_post=False,
        use_abs_posemb=True,
        use_rope2d=True,
        ls_init_value=0.1,
        **kwargs,
    ):
        self.width = width
        self.layers = layers
        self.heads = heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.ues_cls_token = ues_cls_token
        self.use_ln_pre = use_ln_pre
        self.ls_init_value = ls_init_value
        self.use_ln_post = use_ln_post
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        super().__init__(**kwargs)



class StepRoboticsConfig(PretrainedConfig):
    model_type = "step_robotics"
    architectures = ["StepVLForConditionalGeneration"]

    def __init__(
        self,
        vision_config: Optional[Union[dict, StepRoboticsVisionEncoderConfig]] = None,
        text_config: Optional[Union[dict, Qwen3Config]] = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 151679,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = StepRoboticsVisionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = StepRoboticsVisionEncoderConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Qwen3Config()
        elif isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)
        self.text_config = text_config

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.hidden_size = text_config.hidden_size
        self.image_token_id = image_token_id
        # Help Auto classes find the correct implementation when saving/loading.
        super().__init__(**kwargs)
