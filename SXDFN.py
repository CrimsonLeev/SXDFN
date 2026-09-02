import timm
import torch
import torch.nn as nn
from calibration import RigidTransform
from diffdrr.utils import se3_exp_map


def convert(#from DiffPose
    transform,
    input_parameterization,
    output_parameterization,
    input_convention=None,
    output_convention=None,
):
    """Convert between representations of SE(3)."""

    # Convert any input parameterization to a RigidTransform
    if input_parameterization == "se3_log_map":
        transform = torch.concat([transform[1], transform[0]], axis=-1)
        matrix = se3_exp_map(transform).transpose(-1, -2)
        transform = RigidTransform(
            R=matrix[..., :3, :3],
            t=matrix[..., :3, 3],
            device=matrix.device,
            dtype=matrix.dtype,
        )
    elif input_parameterization == "se3_exp_map":
        pass
    else:
        transform = RigidTransform(
            R=transform[0],
            t=transform[1],
            parameterization=input_parameterization,
            convention=input_convention,
        )

    # Convert the RigidTransform to any output
    if output_parameterization == "se3_exp_map":
        return transform
    elif output_parameterization == "se3_log_map":
        se3_log = transform.get_se3_log()
        log_t_vee = se3_log[..., :3]
        log_R_vee = se3_log[..., 3:]
        return log_R_vee, log_t_vee
    else:
        return (
            transform.get_rotation(output_parameterization, output_convention),
            transform.get_translation(),
        )
    
class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(inplace=True))

        super().__init__(*layers)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        hidden_channel = max(channel // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)

        return x * y


class _SXDFNBase(nn.Module):
    """Shared architecture for SXDIN and SXDIN_test."""

    def __init__(
        self,
        model_name="resnet18",
        n_angular_components=3,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.convention = convention

        self.ca1 = SEBlock(
            channel=256,
            reduction=16,
        )
        self.ca2 = SEBlock(
            channel=128,
            reduction=16,
        )
        self.ca3 = SEBlock(
            channel=64,
            reduction=16,
        )
        self.ca4 = SEBlock(
            channel=32,
            reduction=16,
        )

        self.cnn1 = Conv2dReLU(
            256,
            128,
            1,
            0,
            1,
        )
        self.cnn2 = Conv2dReLU(
            128,
            64,
            1,
            0,
            1,
        )
        self.cnn3 = Conv2dReLU(
            64,
            32,
            1,
            0,
            1,
        )

        self.res1 = Conv2dReLU(
            256,
            128,
            1,
            0,
            1,
        )
        self.res2 = Conv2dReLU(
            128,
            64,
            1,
            0,
            1,
        )
        self.res3 = Conv2dReLU(
            64,
            32,
            1,
            0,
            1,
        )

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=33,
            **kwargs,
        )

        output = self._get_backbone_output_dim()

        self.xyz_regression = nn.Linear(
            output,
            3,
        )

        self.rot_regression = nn.Linear(
            output,
            n_angular_components,
        )

    def _get_backbone_output_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                33,
                256,
                256,
            )
            output = self.backbone(dummy)

        return output.shape[-1]

    def _process_y(self, y):
        y_res = self.res1(y)

        y = self.ca1(y)
        y = self.cnn1(y)

        y_res2 = self.res2(y)

        y = self.ca2(y) + y_res
        y = self.cnn2(y)

        y_res3 = self.res3(y)

        y = self.ca3(y) + y_res2
        y = self.cnn3(y)

        y = self.ca4(y) + y_res3

        return y

    def _predict_pose(self, x):
        x = self.backbone(x)

        rot = self.rot_regression(x)
        xyz = self.xyz_regression(x)

        return rot, xyz

    def _convert_pose(self, rot, xyz):
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def forward_features(self, x, y):
        y = self._process_y(y)
        x = torch.cat([x, y], dim=1)

        rot, xyz = self._predict_pose(x)

        return rot, xyz, y


class SXDFN(_SXDFNBase):
    def __init__(
        self,
        model_name="resnet18",
        n_angular_components=3,
        parameterization="se3_log_map",
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            n_angular_components=n_angular_components,
            convention=convention,
            pretrained=pretrained,
            **kwargs,
        )

        self.parameterization = parameterization

    def forward(self, x, y):
        rot, xyz, _ = self.forward_features(x, y)

        return self._convert_pose(rot, xyz)


class SXDFN_test(_SXDFNBase):
    def __init__(
        self,
        model_name="resnet18",
        n_angular_components = 3,
        parameterization="se3_log_map",
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            n_angular_components=n_angular_components,
            convention=convention,
            pretrained=pretrained,
            **kwargs,
        )

        self.parameterization = parameterization

    def forward(self, x, y):
        rot, xyz, y = self.forward_features(x, y)

        pose = self._convert_pose(rot, xyz)

        return pose, y, rot, xyz



、
