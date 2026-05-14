import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)
from PIL import Image
from waveprop.devices import SensorParam, sensor_dict
from tqdm import tqdm

# 並列処理用のライブラリ
from concurrent.futures import ProcessPoolExecutor
import functools

# パス設定
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fftlayer_diff_original import FFTLayer_diff
from config_diffusercam import fft_args

# --- グローバル変数 (Workerプロセス内で初期化されます) ---
simulator_global = None
fft_global = None

# --- 設定値 ---
PADD_SIZE = 270 * 3, 480 * 3
SIZE = 270, 480
sensor = dict(size=np.array([4.8e-6 * 1080, 4.8e-6 * 1920]))
sensor_key = 'poop'
sensor_dict[sensor_key] = sensor


def transform(image, gray=False):
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    # center padding
    image = F.pad(
        image,
        (PADD_SIZE[1] // 2, PADD_SIZE[1] // 2, PADD_SIZE[0] // 2, PADD_SIZE[0] // 2),
        mode='constant',
        value=0,
    )
    # average the RGB channels
    image = image.mean(0, keepdim=True)
    return image


def load_psf(path):
    psf = np.array(Image.open(path))
    return transform(psf)


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate the lensless capture')
    parser.add_argument('--psf_path', default="data/diffusercam/psf.tiff", help='psf folder path')
    parser.add_argument('--obj_path', default="data/diffusercam/ground_truth_lensed", help='object folder path')
    parser.add_argument('--save_path', default="data/diffusercam/decode_sim_padding_png", help='save folder path')
    parser.add_argument('--adj', help='whether to adjust the light intensity', action='store_true', default=False)
    # 並列化用の引数を追加
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='number of parallel workers')

    args = parser.parse_args()
    return args


def init_worker(psf_path, fft_args_config):
    """
    各プロセス（ワーカー）が起動したときに一度だけ呼ばれる初期化関数。
    シミュレーターとFFTモデルをプロセスごとにメモリにロードします。
    """
    global simulator_global, fft_global

    # FFTモデルの初期化
    fft_global = FFTLayer_diff(fft_args_config)

    # PSFのロード
    psf = load_psf(psf_path)

    # シミュレーターの初期化
    # 注意: FarFieldSimulatorのインポートが必要です
    from waveprop.simulation import FarFieldSimulator
    simulator_global = FarFieldSimulator(
        object_height=0.4,
        scene2mask=0.0868 * 4,
        mask2sensor=2e-3,
        sensor=sensor_key,
        psf=psf,
        is_torch=True,
        quantize=False,
        return_float=True
    )

    # PyTorchのマルチスレッド設定（CPU並列化時の競合を防ぐため各プロセスは1スレッドに制限）
    torch.set_num_threads(1)


def process_single_file(file_info):
    """
    1つのファイルを処理する関数
    """
    obj_path_i, save_path, use_adjust_light_intensity = file_info

    try:
        # グローバル変数を使用
        global simulator_global, fft_global

        # load object
        obj = np.load(obj_path_i)
        obj = cv2.normalize(obj, None, 0, 255, cv2.NORM_MINMAX)

        # simulate
        obj = torch.tensor(obj).permute(2, 0, 1).unsqueeze(0).float()

        # Simulator propagate
        img = simulator_global.propagate(obj)

        capture = img
        # Decode
        decoded = fft_global(capture)
        decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())

        decoded = decoded.squeeze().permute(1, 2, 0).detach().numpy()

        decoded = (decoded * 255).astype(np.uint8)

        # 保存パスの生成
        save_png_path = os.path.join(save_path + "_png", os.path.basename(obj_path_i))
        save_png_path = save_png_path.replace(".npy", ".png")
        os.makedirs(os.path.dirname(save_png_path), exist_ok=True)

        cv2.imwrite(save_png_path, decoded)
        return True

    except Exception as e:
        print(f"Error processing {obj_path_i}: {e}")
        return False


def main():
    args = parse_args()
    psf_path = args.psf_path
    obj_path = args.obj_path
    save_path = args.save_path
    adj = args.adj
    num_workers = args.workers

    # ファイルリストの作成
    obj_path_list = []
    if os.path.isdir(obj_path):
        files = os.listdir(obj_path)
        obj_path_list = [os.path.join(obj_path, f) for f in files if f.endswith(".npy")]
    else:
        obj_path_list = [obj_path]

    print(f"Processing {len(obj_path_list)} files with {num_workers} workers...")

    # タスク引数のリスト作成
    tasks = [(f, save_path, adj) for f in obj_path_list]

    # ProcessPoolExecutorによる並列処理
    # initializerを使って各プロセスでモデルをロードすることで、pickleエラーを防ぎ高速化します
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker,
                             initargs=(psf_path, fft_args)) as executor:
        # tqdmで進捗表示
        results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks)))


if __name__ == "__main__":
    main()