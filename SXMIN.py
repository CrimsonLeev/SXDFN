import timm
import torch
import torch.nn as nn
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
            use_batchnorm=False,
    ):
        conv = nn.Conv2d(

            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups = 1
            #bias=not (use_batchnorm),
        )
        relu = nn.LeakyReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv,bn, relu)





class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出尺寸为 (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # squeeze: 全局平均池化
        y = self.avg_pool(x).view(b, c)   # shape: (B, C)
        # excitation: 两个全连接层 + Sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        # scale: 通道加权
        return x * y.expand_as(x)

class SXDIN(torch.nn.Module):
    def __init__(
        self,
        model_name = "renet18",
        n_angular_components = 3,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.convention = convention
        
        self.ca1 = SEBlock(channel=256, reduction=16)
        self.ca2 = SEBlock(channel=128, reduction=16)
        self.ca3 = SEBlock(channel=64, reduction=16)
        self.ca4 = SEBlock(channel=32, reduction=16)
        self.cnn1 = Conv2dReLU(256,128,1,0,1)
        self.cnn2 = Conv2dReLU(128, 64, 1, 0, 1)
        self.cnn3 = Conv2dReLU(64, 32, 1, 0, 1)

        self.res1 = Conv2dReLU(256,128,1,0,1)
        self.res2 = Conv2dReLU(128, 64, 1, 0, 1)
        self.res3 = Conv2dReLU(64, 32, 1, 0, 1)

        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=33,
            **kwargs,
        )

        output = self.backbone(torch.randn(1, 33, 256, 256)).shape[-1]
        self.xyz_regression = torch.nn.Linear(output, 3)
        self.rot_regression = torch.nn.Linear(output, n_angular_components)

    def forward(self, x, y):
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
        #y = self.cnn5(y)
        x = torch.cat([x, y],dim = 1)
        x = self.backbone(x)
        rot = self.rot_regression(x)
        xyz = self.xyz_regression(x)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

class SXDIN_test(torch.nn.Module):

    def __init__(
        self,
        model_name,
        n_angular_components,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.convention = convention

        self.ca1 = SEBlock(channel=256, reduction=16)
        self.ca2 = SEBlock(channel=128, reduction=16)
        self.ca3 = SEBlock(channel=64, reduction=16)
        self.ca4 = SEBlock(channel=32, reduction=16)
        self.cnn1 = Conv2dReLU(256,128,1,0,1)
        self.cnn2 = Conv2dReLU(128, 64, 1, 0, 1)
        self.cnn3 = Conv2dReLU(64, 32, 1, 0, 1)

        self.res1 = Conv2dReLU(256,128,1,0,1)
        self.res2 = Conv2dReLU(128, 64, 1, 0, 1)
        self.res3 = Conv2dReLU(64, 32, 1, 0, 1)
        #self.cnn5 = Conv2dReLU(8, 3, 3, 1, 1)
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=33,
            **kwargs,
        )

        output = self.backbone(torch.randn(1, 33, 256, 256)).shape[-1]
        print(self.backbone(torch.randn(1, 33, 256, 256)).shape)
        self.xyz_regression = torch.nn.Linear(output, 3)
        self.rot_regression = torch.nn.Linear(output, n_angular_components)

    def forward(self, x, y):
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
        #y = self.cnn5(y)
        x = torch.cat([x, y],dim = 1)
        x = self.backbone(x)
        rot = self.rot_regression(x)
        xyz = self.xyz_regression(x)
        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        ),y,rot,xyz



、
