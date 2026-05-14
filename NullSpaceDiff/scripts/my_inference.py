"""make variations of input image"""
import argparse, os, sys, glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
# timeは削除し、torch.cudaの機能を使用しますが、全体の初期化等で使う可能性があるので残してもOKです
import time
from pytorch_lightning import seed_everything

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from scripts.helper import DDNMGuidance

# LPIPSライブラリ
import lpips

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
       if section_counts.startswith("ddim"):
          desired_count = int(section_counts[len("ddim"):])
          for i in range(1, num_timesteps):
             if len(range(0, num_timesteps, i)) == desired_count:
                return set(range(0, num_timesteps, i))
          raise ValueError(
             f"cannot create exactly {num_timesteps} steps with an integer stride"
          )
       section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
       size = size_per + (1 if i < extra else 0)
       if size < section_count:
          raise ValueError(
             f"cannot divide section of {size} steps into {section_count}"
          )
       if section_count <= 1:
          frac_stride = 1
       else:
          frac_stride = (size - 1) / (section_count - 1)
       cur_idx = 0.0
       taken_steps = []
       for _ in range(section_count):
          taken_steps.append(start_idx + round(cur_idx))
          cur_idx += frac_stride
       all_steps += taken_steps
       start_idx += size
    return set(all_steps)

class ImageDataset(Dataset):
    def __init__(self, init_img_dir, outpath, transform=None, gpu_id=0, gpu_num=1):
        self.init_img_dir = init_img_dir
        self.outpath = outpath
        self.transform = transform if transform else transforms.ToTensor()
        self.img_list = [img for img in sorted(os.listdir(init_img_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.img_list = [img for img in self.img_list if not os.path.exists(os.path.join(outpath, img))][gpu_id::gpu_num]
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.init_img_dir, img_name)
        image = load_img(img_path)[0]
        if self.transform:
            image = self.transform(image)
        image = image.clamp(-1, 1)
        return image, img_name

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
       print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
       print("missing keys:")
       print(m)
    if len(u) > 0 and verbose:
       print("unexpected keys:")
       print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-img", type=str, nargs="?", default="data/flatnet2single/inputs")
    parser.add_argument("--outdir", type=str, nargs="?", default="data/flatnet2single/outputs_ft")
    parser.add_argument("--ddpm_steps", type=int, default=1000)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--config", type=str, default="configs/stableSRNew/v2-finetune_lensless_T_512.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt")
    parser.add_argument("--vqgan_ckpt", type=str, default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--colorfix_type", type=str, default="nofix")

    opt = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
       print('Use adain color correction')
    elif opt.colorfix_type == 'wavelet':
       print('Use wavelet color correction')
    else:
       print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    seed_everything(opt.seed)

    transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize(opt.input_size),
       torchvision.transforms.CenterCrop(opt.input_size),
    ])

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)
    ddnm_guidance = None

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    image_dataset = ImageDataset(opt.init_img, outpath, transform=transform, gpu_id=opt.gpu_id, gpu_num=opt.gpu_num)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size // 2, pin_memory=True)

    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                     linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000

    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
       if i in use_timesteps:
          new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
          last_alpha_cumprod = alpha_cumprod
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)

    # 評価指標用リスト
    mse_list = []
    lpips_list = []
    time_list = []

    print("Loading LPIPS model...")
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    # 【追加】CUDAイベントの初期化
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
       with precision_scope("cuda"):
          with model.ema_scope():
             for init_image, img_names in tqdm(image_dataloader):
                init_image = init_image.to(device)

                # ========== 時間計測開始 (CUDA Event) ==========
                starter.record()
                # ============================================

                init_latent_generator = model.encode_first_stage(init_image)
                init_latent = model.get_first_stage_encoding(init_latent_generator)
                text_init = ['']*init_image.size(0)
                semantic_c = model.cond_stage_model(text_init)

                noise = torch.randn_like(init_latent)
                t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
                t = t.to(device).long()
                x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)

                samples, _ = model.sample(cond=semantic_c,
                ddnm_guidance=ddnm_guidance, struct_cond=init_latent, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)

                x_samples = model.decode_first_stage(samples)
                if opt.colorfix_type == 'adain':
                   x_samples = adaptive_instance_normalization(x_samples, init_image)
                elif opt.colorfix_type == 'wavelet':
                   x_samples = wavelet_reconstruction(x_samples, init_image)

                # ========== 時間計測終了 (CUDA Event) ==========
                ender.record()
                torch.cuda.synchronize() # GPUの処理完了を待機

                # elapsed_timeはミリ秒で返るため、1000で割って秒に変換
                curr_time_ms = starter.elapsed_time(ender)
                current_batch_size = init_image.size(0)
                time_per_image = curr_time_ms / current_batch_size
                time_list.extend([time_per_image] * current_batch_size)
                # ============================================

                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                # ========== 指標計算 ==========
                gt_image_0to1 = (init_image + 1.0) / 2.0

                # MSE
                mse_val = torch.mean((x_samples - gt_image_0to1) ** 2, dim=[1, 2, 3])
                mse_list.extend(mse_val.cpu().numpy().tolist())

                # LPIPS (range: -1 to 1)
                pred_image_lpips = x_samples * 2.0 - 1.0
                gt_image_lpips = init_image
                lpips_val = loss_fn_lpips(pred_image_lpips, gt_image_lpips)
                lpips_list.extend(lpips_val.flatten().cpu().numpy().tolist())
                # ===========================

                for i in range(init_image.size(0)):
                   # img_name = image_dataset.img_list.pop(0)
                   img_name = img_names[i]
                   basename = os.path.splitext(os.path.basename(img_name))[0]
                   x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                   Image.fromarray(x_sample.astype(np.uint8)).save(
                      os.path.join(outpath, basename+'.png'))

    # 結果出力
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    if len(mse_list) > 0:
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list)
        print(f"MSE  : {mse_mean:.6f} +/- {mse_std:.6f}")

        lpips_mean = np.mean(lpips_list)
        lpips_std = np.std(lpips_list)
        print(f"LPIPS: {lpips_mean:.6f} +/- {lpips_std:.6f}")

        time_mean = np.mean(time_list)
        time_std = np.std(time_list)
        print(f"Time : {time_mean:.4f} ms/img +/- {time_std:.4f}")
    else:
        print("No images were processed.")

    print("="*50)
    print(f"Your samples are ready: {outpath}")

if __name__ == "__main__":
    main()