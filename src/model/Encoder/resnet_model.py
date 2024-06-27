from typing import List
from robomimic.models.obs_core import VisualCore

def get_resnet(input_shape: List[int], output_size: int):
    """Get ResNet model from torchvision.models
    Args:
        input_shape: Shape of input image (C, H, W).
        output_size: Size of output feature vector.
    """

    resnet = VisualCore(
        input_shape=input_shape,
        backbone_class="ResNet18Conv",
        backbone_kwargs=dict(
            input_coord_conv=False,
            pretrained=False,
        ),
        pool_class="SpatialSoftmax",
        pool_kwargs=dict(
            num_kp=32,
            learnable_temperature=False,
            temperature=1.0,
            noise_std=0.0,
            output_variance=False,
        ),
        flatten=True,
        feature_dimension=output_size,
    )

    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model
