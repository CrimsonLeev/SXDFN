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
        #print(000)
        #print(transform)
        transform = torch.concat([transform[1], transform[0]], axis=-1)
        #print(transform)
        matrix = se3_exp_map(transform).transpose(-1, -2)
        #print(matrix)
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
        parameterization = "se3_log_map",
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

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
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
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



# %% ../notebooks/api/03_registration.ipynb 6
N_ANGULAR_COMPONENTS = {
    "axis_angle": 3,
    "euler_angles": 3,
    "se3_log_map": 3,
    "quaternion": 4,
    "rotation_6d": 6,
    "rotation_10d": 10,
    "quaternion_adjugate": 10,
}

from diffdrr.detector import make_xrays
from diffdrr.drr import DRR
from diffdrr.siddon import siddon_raycast

from calibration import RigidTransform


class SparseRegistration(torch.nn.Module):
    def __init__(
        self,
        drr: DRR,
        pose: RigidTransform,
        parameterization: str,
        convention: str = None,
        features=None,  # Used to compute biased estimate of mNCC
        n_patches: int = None,  # If n_patches is None, render the whole image
        patch_size: int = 13,
    ):
        super().__init__()
        self.drr = drr

        # Parse the input pose
        rotation, translation = convert(
            pose,
            input_parameterization="se3_exp_map",
            output_parameterization=parameterization,
            output_convention=convention,
        )
        self.parameterization = parameterization
        self.convention = convention
        self.rotation = torch.nn.Parameter(rotation)
        self.translation = torch.nn.Parameter(translation)

        # Crop pixels off the edge such that pixels don't fall outside the image
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.patch_radius = self.patch_size // 2 + 1
        self.height = self.drr.detector.height
        self.width = self.drr.detector.width
        self.f_height = self.height - 2 * self.patch_radius
        self.f_width = self.width - 2 * self.patch_radius

        # Define the distribution over patch centers
        if features is None:
            features = torch.ones(
                self.height, self.width, device=self.rotation.device
            ) / (self.height * self.width)
        self.patch_centers = torch.distributions.categorical.Categorical(
            probs=features.squeeze()[
                self.patch_radius : -self.patch_radius,
                self.patch_radius : -self.patch_radius,
            ].flatten()
        )

    def forward(self, n_patches=None, patch_size = None):
        # Parse initial density
        if not hasattr(self.drr, "density"):
            self.drr.set_bone_attenuation_multiplier(
                self.drr.bone_attenuation_multiplier
            )

        if n_patches is not None or patch_size is not None:
            self.n_patches = n_patches
            self.patch_size = patch_size

        # Make the mask for sparse rendering
        if self.n_patches is None:
            mask = torch.ones(
                1,
                self.height,
                self.width,
                dtype=torch.bool,
                device=self.rotation.device,
            )
        else:
            mask = torch.zeros(
                self.n_patches,
                self.height,
                self.width,
                dtype=torch.bool,
                device=self.rotation.device,
            )
            radius = self.patch_size // 2
            idxs = self.patch_centers.sample(sample_shape=torch.Size([self.n_patches]))
            idxs, jdxs = (
                idxs // self.f_height + self.patch_radius,
                idxs % self.f_width + self.patch_radius,
            )

            idx = torch.arange(-radius, radius + 1, device=self.rotation.device)
            patches = torch.cartesian_prod(idx, idx).expand(self.n_patches, -1, -1)
            patches = patches + torch.stack([idxs, jdxs], dim=-1).unsqueeze(1)
            patches = torch.concat(
                [
                    torch.arange(self.n_patches, device=self.rotation.device)
                    .unsqueeze(-1)
                    .expand(-1, self.patch_size**2)
                    .unsqueeze(-1),
                    patches,
                ],
                dim=-1,
            )
            mask[
                patches[..., 0],
                patches[..., 1],
                patches[..., 2],
            ] = True

        # Get the source and target
        pose = convert(
            [self.rotation, self.translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )
        source, target = make_xrays(
            pose,
            self.drr.detector.source,
            self.drr.detector.target,
        )

        # Render the sparse image
        target = target[mask.any(dim=0).view(1, -1)]
        img = siddon_raycast(source, target, self.drr.density, self.drr.spacing)
        if self.n_patches is None:
            img = self.drr.reshape_transform(img, batch_size=len(self.rotation))

        return img, mask, pose

    def get_current_pose(self):
        return convert(
            [self.rotation, self.translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

class SparseRegistration1(torch.nn.Module):
    def __init__(
        self,
        drr: DRR,
        pose: RigidTransform,
        parameterization: str,
        convention: str = None,
        features=None,  # Used to compute biased estimate of mNCC
        n_patches: int = None,  # If n_patches is None, render the whole image
        patch_size: int = 13,
    ):
        super().__init__()
        self.drr = drr

        # Parse the input pose
        rotation, translation = convert(#将模型得到的变换参数转换回旋转平移参数
            pose,
            input_parameterization="se3_exp_map",
            output_parameterization=parameterization,
            output_convention=convention,
        )
        self.parameterization = parameterization
        self.convention = convention
        self.rotation = torch.nn.Parameter(rotation)#以se3的空间转换旋转参数
        self.translation = torch.nn.Parameter(translation)#以se3的空间转换平移参数

        # Crop pixels off the edge such that pixels don't fall outside the image
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.patch_radius = self.patch_size // 2 + 1
        self.height = self.drr.detector.height
        self.width = self.drr.detector.width
        self.f_height = self.height - 2 * self.patch_radius
        self.f_width = self.width - 2 * self.patch_radius

        # Define the distribution over patch centers
        if features is None:
            features = torch.ones(
                self.height, self.width, device=self.rotation.device
            ) / (self.height * self.width)
        self.patch_centers = torch.distributions.categorical.Categorical(
            probs=features.squeeze()[
                self.patch_radius : -self.patch_radius,
                self.patch_radius : -self.patch_radius,
            ].flatten()
        )

    def forward(self, n_patches=100, patch_size=13):
        # Parse initial density
        if not hasattr(self.drr, "density"):
            self.drr.set_bone_attenuation_multiplier(
                self.drr.bone_attenuation_multiplier
            )

        if n_patches is not None or patch_size is not None:
            self.n_patches = n_patches
            self.patch_size = patch_size

        # Make the mask for sparse rendering
        if self.n_patches is None:
            mask = torch.ones(
                1,
                self.height,
                self.width,
                dtype=torch.bool,
                device=self.rotation.device,
            )
        else:
            mask = torch.zeros(
                self.n_patches,
                self.height,
                self.width,
                dtype=torch.bool,
                device=self.rotation.device,
            )
            radius = self.patch_size // 2
            idxs = self.patch_centers.sample(sample_shape=torch.Size([self.n_patches]))
            idxs, jdxs = (
                idxs // self.f_height + self.patch_radius,
                idxs % self.f_width + self.patch_radius,
            )

            idx = torch.arange(-radius, radius + 1, device=self.rotation.device)
            patches = torch.cartesian_prod(idx, idx).expand(self.n_patches, -1, -1)
            patches = patches + torch.stack([idxs, jdxs], dim=-1).unsqueeze(1)
            patches = torch.concat(
                [
                    torch.arange(self.n_patches, device=self.rotation.device)
                    .unsqueeze(-1)
                    .expand(-1, self.patch_size**2)
                    .unsqueeze(-1),
                    patches,
                ],
                dim=-1,
            )
            mask[
                patches[..., 0],
                patches[..., 1],
                patches[..., 2],
            ] = True

        # Get the source and target
        pose = convert(
            [self.rotation, self.translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )
        source, target = make_xrays(
            pose,
            self.drr.detector.source,
            self.drr.detector.target,
        )

        # Render the sparse image
        target = target[mask.any(dim=0).view(1, -1)]
        img = siddon_raycast(source, target, self.drr.density, self.drr.spacing)

        if self.n_patches is None:
            img = self.drr.reshape_transform(img, batch_size=len(self.rotation))
        return img, mask

    def get_current_pose(self):
        return convert(
            [self.rotation, self.translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

def preprocess(x, eps=1e-4):
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return (x - 0.3080) / 0.1494


def pred_to_patches(pred_img, mask, n_patches, patch_size):
    return pred_img.expand(-1, n_patches, -1)[..., mask[..., mask.any(dim=0)]].reshape(
        1, n_patches, -1
    )


def img_to_patches(img, mask, n_patches, patch_size):
    return img.expand(-1, n_patches, -1, -1)[..., mask].reshape(1, n_patches, -1)


def mask_to_img(img, mask):
    return img[..., mask.any(dim=0)]


class VectorizedNormalizedCrossCorrelation2d(torch.nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, img, pred_img, mask, n_patches, patch_size):
        pred_img = preprocess(pred_img).unsqueeze(0)
        sub_img = mask_to_img(img, mask)
        pred_patches = pred_to_patches(pred_img, mask, n_patches, patch_size)
        img_patches = img_to_patches(img, mask, n_patches, patch_size)

        local_ncc = self.forward_compute(pred_patches, img_patches)
        global_ncc = self.forward_compute(pred_img, sub_img)
        return (local_ncc + global_ncc) / 2

    def forward_compute(self, x1, x2):
        assert x1.shape == x2.shape, "Input images must be the same size"
        x1, x2 = self.norm(x1), self.norm(x2)
        ncc = (x1 * x2).mean(dim=[-1, -2])
        return ncc

    def norm(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0) + self.eps
        std = var.sqrt()
        return (x - mu) / std
