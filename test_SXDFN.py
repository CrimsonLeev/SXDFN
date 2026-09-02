import time
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from calibration import convert
from deepfluoro import DeepFluoroDataset,Evaluator,Transforms
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from metrics import DoubleGeodesic, GeodesicSE3
from SXDFN import SXDFN_test

# ============================================================
# Configuration
# ============================================================
@dataclass
class RegistrationConfig:
    # Data
    patient_id: int = 1
    # Image resolution
    high_resolution: int = 256
    low_resolution: int = 64
    # CT preprocessing
    ct_target_size: int = 512
    ct_scale: float = 1.25
    # Registration
    n_iters: int = 100
    low_resolution_iters: int = 80
    rotation_lr: float = 3e-2
    translation_lr: float = 3e0
    # NCC
    ncc_scales: tuple = (None, 13)
    ncc_weights: tuple = (0.5, 0.5)
    lowres_ncc_scales: tuple = (None, 7)
    lowres_ncc_weights: tuple = (0.5, 0.5)
    # Model
    checkpoint: str = "checkpoints/SXMIN/fine_01_best.ckpt"
    norm_layer: str = "groupnorm"
    # Misc
    device: str = "cuda"
    verbose: bool = False
    display: bool = False

# ============================================================
# CT preprocessing
# ============================================================
def process_volume(volume, device, target_size=512, scale=1.25):
    """Preprocess CT volume for SXMIN."""
    volume = torch.as_tensor(volume, dtype=torch.float32)
    d, h, w = volume.shape # Original CT size
    new_size = (int(scale * d), int(scale * h), int(scale * w))
    volume = F.interpolate(volume[None, None], size=new_size, mode="trilinear", align_corners=True)
    _, _, d, h, w = volume.shape
    if d > target_size or h > target_size or w > target_size:
        raise ValueError(f"Resized CT shape {(d, h, w)} exceeds target size {target_size}.")
    # Center padding with -1000 HU
    pad_d, pad_h, pad_w = target_size - d, target_size - h, target_size - w
    padding = (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2, pad_d//2, pad_d-pad_d//2)
    volume = F.pad(volume, padding, mode="constant", value=-1000)
    volume = F.max_pool3d(volume, kernel_size=2, stride=2) # Downsample 2x
    volume = volume.squeeze(0) # Remove batch dimension
    volume = volume.permute(0, 2, 1, 3) # Rearrange dimensions
    volume = TF.rotate(volume, angle=90) # Rotate 90 degrees
    return volume.to(device=device, dtype=torch.float32)

# ============================================================
# DRR construction
# ============================================================
def build_drr(specimen, height):
    """Construct differentiable DRR for a specific resolution."""
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    return DRR(specimen.volume, specimen.spacing, sdr=specimen.focal_len / 2,
               height=height, delx=delx, x0=specimen.x0, y0=specimen.y0,
               reverse_x_axis=True, bone_attenuation_multiplier=2.5)

# ============================================================
# Model
# ============================================================
def load_model(config: RegistrationConfig, device: torch.device):
    """Build SXMIN_test and load pretrained weights."""
    model = SXDFN_test(norm_layer=config.norm_layer).to(device)
    checkpoint = torch.load(config.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {config.checkpoint}")
    return model

# ============================================================
# Registration
# ============================================================
class Registration:
    def __init__(self, specimen, model, drr_high, drr_low, config, device):
        self.specimen, self.model, self.config, self.device = specimen, model, config, device
        self.drr_high, self.drr_low = drr_high.to(device), drr_low.to(device)
        self.ct = process_volume(specimen.volume, device=device, target_size=config.ct_target_size, scale=config.ct_scale)
        self.isocenter_pose = specimen.isocenter_pose.to(device)
        self.transforms_high = Transforms(self.drr_high.detector.height)
        self.transforms_low = Transforms(self.drr_low.detector.height)
        self.transforms_display = Transforms(1436) # Used for fiducial visualization
        self.geodesic, self.double_geodesic = GeodesicSE3(), DoubleGeodesic(sdr=specimen.focal_len / 2)
        self.criterion_low = MultiscaleNormalizedCrossCorrelation2d(list(config.lowres_ncc_scales), list(config.lowres_ncc_weights))
        self.criterion_high = MultiscaleNormalizedCrossCorrelation2d(list(config.ncc_scales), list(config.ncc_weights))
        self.evaluator = None

    @torch.no_grad()
    def predict_initial_pose(self, image):
        """Obtain the initial pose from SXMIN_test."""
        self.model.eval()
        pred_pose, _, rotation, translation = self.model(image, self.ct)
        return rotation, translation

    def refine_pose(self, image_high, image_low, rotation, translation):
        """Iteratively refine the pose, multi‑resolution scheme."""
        config = self.config
        rotation = nn.Parameter(rotation.detach().clone())
        translation = nn.Parameter(translation.detach().clone())
        optimizer = torch.optim.Adam([{"params":[rotation],"lr":config.rotation_lr},{"params":[translation],"lr":config.translation_lr}], maximize=True)
        final_pose = None
        for iteration in range(config.n_iters):
            optimizer.zero_grad(set_to_none=True)
            # Convert parameters to SE(3)
            pred_pose = convert([rotation, translation],
                input_parameterization=config.parameterization if hasattr(config,"parameterization") else None,
                output_parameterization="se3_exp_map",
                input_convention=config.convention if hasattr(config,"convention") else None)
            # Multi‑resolution refinement
            if iteration < config.low_resolution_iters:
                pred_img = self.drr_low(None, None, None, pose=pred_pose)
                loss = self.criterion_low(image_low, pred_img)
            else:
                pred_img = self.drr_high(None, None, None, pose=pred_pose)
                loss = self.criterion_high(image_high, pred_img)
            loss.backward()
            optimizer.step()
            final_pose = pred_pose
        return final_pose

    def evaluate(self, idx, pred_pose):
        """Calculate target registration error."""
        evaluator = Evaluator(self.specimen, idx)
        tre = evaluator(pred_pose.cpu())
        return tre.item() if torch.is_tensor(tre) else tre

    def visualize(self, idx, reference_img, pred_pose, tre):
        """Draw registration results, fiducials and difference map."""
        pred_img = self.drr_high(None, None, None, pose=pred_pose)
        pred_img = self.transforms_high(pred_img)
        reference_np = reference_img.detach().cpu().numpy().squeeze()
        pred_np = pred_img.detach().cpu().numpy().squeeze()
        # Reference vs prediction
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(reference_np, cmap="gray");axes[0].set_title("Reference");axes[0].axis("off")
        axes[1].imshow(pred_np, cmap="gray");axes[1].set_title(f"TRE: {tre:.4f} mm");axes[1].axis("off")
        plt.tight_layout();plt.show()
        # Fiducials
        true_fiducials, pred_fiducials = self.specimen.get_2d_fiducials(idx, pred_pose)
        display_img = self.transforms_display(pred_img)
        display_np = display_img.detach().cpu().numpy().squeeze()
        plt.figure(figsize=(8, 8));plt.imshow(display_np, cmap="gray")
        plt.scatter(true_fiducials[0,...,0], true_fiducials[0,...,1], label="True Fiducials")
        plt.scatter(pred_fiducials.detach().cpu().numpy()[0,...,0], pred_fiducials.detach().cpu().numpy()[0,...,1], marker="x", label="Predicted Fiducials")
        for fiducial_id in range(true_fiducials.shape[1]):
            plt.plot([true_fiducials[...,fiducial_id,0].item(),pred_fiducials[...,fiducial_id,0].item()],
                     [true_fiducials[...,fiducial_id,1].item(),pred_fiducials[...,fiducial_id,1].item()],"--")
        plt.title("Fiducial Projection");plt.legend();plt.axis("off");plt.show()
        # Difference map
        difference = reference_np - pred_np
        plt.figure(figsize=(8, 8));plt.imshow(difference, cmap="bwr", vmin=-1, vmax=1)
        plt.colorbar();plt.title("Difference Map");plt.axis("off");plt.show()

    def run(self, idx):
        """Register one sample image."""
        image, target_pose = self.specimen[idx]
        image_high, image_low = self.transforms_high(image).to(self.device), self.transforms_low(image).to(self.device)
        target_pose = target_pose.to(self.device)
        rotation, translation = self.predict_initial_pose(image_high)
        start_time = time.perf_counter()
        pred_pose = self.refine_pose(image_high=image_high, image_low=image_low, rotation=rotation, translation=translation)
        elapsed_time = time.perf_counter() - start_time
        tre = self.evaluate(idx, pred_pose)
        if self.config.display: self.visualize(idx=idx, reference_img=image_high, pred_pose=pred_pose, tre=tre)
        return tre, elapsed_time

# ============================================================
# Dataset evaluation
# ============================================================
def evaluate_dataset(registration):
    """Evaluate all images of one specimen."""
    specimen = registration.specimen
    tre_list, time_list = [], []
    success_10, success_2, success_1 = 0, 0, 0
    n_samples = len(specimen)
    for idx in tqdm(range(n_samples), ncols=100, desc="Evaluation"):
        tre, elapsed_time = registration.run(idx)
        tre_list.append(tre);time_list.append(elapsed_time)
        if tre < 10: success_10 += 1
        if tre < 2: success_2 += 1
        if tre < 1: success_1 += 1
        print(f"Sample {idx:03d} | TRE: {tre:.4f} mm | Time: {elapsed_time:.3f} s")
    # Statistics
    tre_array, time_array = np.asarray(tre_list, dtype=np.float64), np.asarray(time_list, dtype=np.float64)
    mean_tre, std_tre, mean_time = tre_array.mean(), tre_array.std(), time_array.mean()
    sms_rate, success_rate, success_rate_10 = success_1/n_samples, success_2/n_samples, success_10/n_samples
    print("\n" + "=" * 60);print("Evaluation Results");print("=" * 60)
    print(f"mTRE  : {mean_tre:.4f} ± {std_tre:.4f} mm")
    print(f"SMSR  : {sms_rate:.4f}");print(f"SR    : {success_rate:.4f}");print(f"SR10  : {success_rate_10:.4f}")
    print(f"Time  : {mean_time:.4f} s/image");print("=" * 60)
    return {"mTRE": mean_tre, "std_TRE": std_tre, "SMSR": sms_rate, "SR": success_rate, "SR10": success_rate_10, "mean_time": mean_time, "TRE": tre_array, "time": time_array}

# ============================================================
# Main
# ============================================================
def main(patient_id=1, parameterization=None, convention=None):
    config = RegistrationConfig(patient_id=patient_id)
    # Device
    device = torch.device("cuda") if (config.device=="cuda" and torch.cuda.is_available()) else torch.device("cpu")
    print(f"Using device: {device}")
    specimen = DeepFluoroDataset(config.patient_id)
    drr_high, drr_low = build_drr(specimen,config.high_resolution), build_drr(specimen, config.low_resolution)
    model = load_model(config, device)
    registration = Registration(specimen=specimen, model=model, drr_high=drr_high, drr_low=drr_low, config=config, device=device)
    results = evaluate_dataset(registration)
    return results

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main(patient_id=1, parameterization="se3_log_map", convention=None)


    main(id_number=1,n_angular_components = 3)


