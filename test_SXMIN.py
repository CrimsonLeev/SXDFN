import time

import torch
#from drr.drr3 import DRR
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from torchvision.transforms.functional import resize

from tqdm import tqdm
import torch.nn.functional as F
from calibration import convert
from deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from metrics import DoubleGeodesic, GeodesicSE3
from SXMIN import SXMIN_test
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np


def process_volume(volume,device):
    volume = torch.from_numpy((volume))
    down = nn.MaxPool3d(2)
    new_D = int(1.25 * volume.shape[0])
    new_H = int(1.25 * volume.shape[1])
    new_W = int(1.25 * volume.shape[2])
    resize_volume = F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=(new_D, new_H, new_W), mode='trilinear',
                             align_corners=True)
    padding = (
        (512 - new_W) // 2,
        512 - new_W - ((512 - new_W) // 2),
        (512 - new_H) // 2,
        512 - new_H - ((512 - new_H) // 2),
        (512 - new_D) // 2,
        512 - new_D - ((512 - new_D) // 2),
    )

    i_ct = F.pad(resize_volume, padding, "constant", -1000).squeeze(0)
    i_ct = down(i_ct).to(device, dtype=torch.float)
    i_ct = i_ct.permute(0, 2, 1, 3)

    #
    i_ct = T.functional.rotate(i_ct, angle=90)
    return i_ct
class Registration:
    def __init__(
        self,
        drr,#DRR method used for 256*256 rendering
        drr2,#DRR method used for 128*128 rendering
        specimen,
        model,
        parameterization,
        convention=None,
        n_iters=0,
        verbose=False,
        device="cuda",
        display = False
    ):
        self.device = torch.device(device)
        self.drr = drr.to(self.device)
        self.drr2 = drr2.to(self.device)
        self.model = model.to(self.device)
        self.display = display
        v_ct = specimen.volume
        v_ct = torch.from_numpy((v_ct))
        #
        self.imgx = process_volume(v_ct,device)

        self.specimen = specimen
        self.isocenter_pose = specimen.isocenter_pose.to(self.device)

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.focal_len / 2)
        self.criterion0 = MultiscaleNormalizedCrossCorrelation2d([None, 7], [0.5, 0.5])
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])

        self.transforms = Transforms(self.drr.detector.height)
        self.transforms2 = Transforms(self.drr2.detector.height)
        self.transforms3 = Transforms(1436)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose
        self.down = nn.MaxPool2d(2)



    def evaluate2(self, pred_pose):
        est_pose = pred_pose
        rot = est_pose.get_rotation("euler_angles", "ZYX")
        xyz = est_pose.get_translation()
        alpha, beta, gamma = rot.squeeze().tolist()
        bx, by, bz = xyz.squeeze().tolist()
        tre = self.target_registration_error(est_pose.cpu()).item()
        return tre

    def run(self, idx):
        imga, pose = self.specimen[idx]
        img = self.transforms(imga).to(self.device)
        imgb = self.transforms2(imga).to(self.device)

        self.pose = pose.to(self.device)
        img_drr = self.drr(None, None, None, pose=self.pose)
        img_drr = self.transforms(img_drr).to(self.device)

        self.model.train()
        pred_pose, y, rotation, translation = self.model(img, self.imgx)

        rotation = torch.nn.Parameter(rotation)#.to(self.device)
        translation = torch.nn.Parameter(translation)#.to(self.device)
        optimizer = torch.optim.Adam(
            [
                {"params": [rotation], "lr": 3e-2},
                {"params": [translation], "lr": 3e0
                 },
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=25,
            gamma=0.9,
        )
        self.target_registration_error = Evaluator(self.specimen, idx)
        losses = []
        times = []
        itr = (
            tqdm(range(self.n_iters), ncols=75) if self.verbose else range(self.n_iters)
        )


        s = time.time()
        for i in itr:

            optimizer.zero_grad()
            pred_pose = convert(
            [rotation, translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
            )

            if i < 80: #Iteration under low resolution
                pred_img = self.drr2(None, None, None, pose=pred_pose)
                loss = self.criterion0(imgb,pred_img)
            else:
                pred_img = self.drr(None, None, None, pose=pred_pose)
                loss = self.criterion0(img, pred_img)
            loss.backward()
            optimizer.step()

        e = time.time()
        cost = e-s

        pred_img = self.drr(None, None, None, pose=pred_pose)
        pred_img = self.transforms(pred_img).to(self.device)
        pred_img2 = self.transforms3(pred_img)
        posea = pred_pose.to('cpu')

        tre = self.evaluate2(posea)

        if self.display:

            img_np = img_drr.to('cpu').detach().numpy().squeeze()
            pred_img_np = pred_img.to('cpu').detach().numpy().squeeze()
            pred_img_np2 = pred_img2.to('cpu').detach().numpy().squeeze()
            # create a figure with two tab
            fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 

            # display img in the first tab
            axs[0].imshow(img_np, cmap="gray")
            axs[0].set_title("Refer")
            axs[0].axis('off')

            # display pred_img in the seconde tab
            axs[1].imshow(pred_img_np, cmap="gray")
            axs[1].set_title(tre)  
            axs[1].axis('off')

            plt.show()

            true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, pred_pose)
            plt.figure(constrained_layout=True)
            plt.title("DRR")
            plt.imshow(pred_img_np2, cmap="gray")
            plt.scatter(
                true_fiducials[0, ..., 0],
                true_fiducials[0, ..., 1],
                label="True Fiducials",
            )
            plt.scatter(
                pred_fiducials.detach().numpy()[0, ..., 0],
                pred_fiducials.detach().numpy()[0, ..., 1],
                marker="x",
                c="tab:orange",
                label="Predicted Fiducials",
            )
            for idxx in range(true_fiducials.shape[1]):
                plt.plot(
                    [true_fiducials[..., idxx, 0].item(), pred_fiducials[..., idxx, 0].item()],
                    [true_fiducials[..., idxx, 1].item(), pred_fiducials[..., idxx, 1].item()],
                    "w--",
                )
            plt.legend()
            plt.axis('off')
            plt.show()

            difference = img_np - pred_img_np

            # display error map
            plt.imshow(difference, cmap='bwr', vmin=-1, vmax=1)
            plt.colorbar()
            plt.show()

        loss = self.criterion(img, pred_img)#, mask, n_patches=50, patch_size=13)
        losses.append(loss.item())
        times.append(0)
        return tre, cost

def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/SXMIN/fine_01_best.ckpt", map_location=torch.device('cuda'))
    model = SXMIN_test(norm_layer="groupnorm")
    model.load_state_dict(ckpt)

    specimen = DeepFluoroDataset(id_number)
    height = 256
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    )
    drr2 = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=64,
        delx=delx * 4,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    )

    registration = Registration(
        drr,
        drr2,
        specimen,
        model,
        parameterization,
    )
    mTRE = []
    sum_num = len(specimen)
    su = 0
    suc = 0
    suc1 = 0

    for idx in tqdm(range(len(specimen)), ncols=100):
        tre, cost = registration.run(idx)
        print("TRE: ",tre," Time: ",cost)
        if tre < 10:
            su = su + 1
            if tre < 2:
                suc1 = suc1 + 1
                if tre < 1:
                    suc = suc + 1
        mTRE.append(tre)

    mTRE_array = np.array(mTRE)
    mean_value = mTRE_array.mean()
    variance = mTRE_array.std()
    print("mTRE: ", mean_value, " +- ", variance)
    print("SMSR: ", suc / sum_num)
    print("SR: ", suc1 / sum_num)
    print("SR10:",su / sum_num)


if __name__ == "__main__":

    main(id_number=1,n_angular_components = 3)


