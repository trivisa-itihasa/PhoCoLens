# process svd output to dataset that can be used by NullSpaceDiff   
# first, move the output to a single folder
# second, move the original images to a single folder
# third, resize the images to 512x512

#find the directory of the process file
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
tools_dir = os.path.dirname(os.path.abspath(__file__))
svd_dir = os.path.join(os.path.dirname(tools_dir), 'SVDeconv')
nullspace_dir = os.path.join(os.path.dirname(tools_dir), 'NullSpaceDiff')
dataset = "diffusercam"

exp_name = "diff"
output_name = exp_name
output_val_dir = os.path.join(nullspace_dir, "data/%s/%s/val"%(dataset,output_name))
output_train_dir = os.path.join(nullspace_dir, "data/%s/%s/train"%(dataset,output_name))
source_svd_train_dir = os.path.join(svd_dir, 'output/%s'%dataset, exp_name, "train")
source_svd_val_dir = os.path.join(svd_dir, 'output/%s'%dataset, exp_name, "val")
source_orig_dir = os.path.join(svd_dir, "data/%s/orig"%dataset)

# index = 0
# for cls in tqdm(os.listdir(source_svd_train_dir)):
#     cls_dir = os.path.join(source_svd_train_dir, cls)
#     if not os.path.isdir(cls_dir):
#         continue
#     for file in os.listdir(cls_dir):
#         if file.endswith('png') and file.startswith('output_'):
#             os.makedirs(os.path.join(output_train_dir, "inputs"), exist_ok=True)
#             shutil.copy(os.path.join(cls_dir, file), os.path.join(os.path.join(output_train_dir, "inputs"), file[7:]))
#             gt_file = os.path.join(source_orig_dir, cls, file[7:]).replace('png', 'JPEG')
#             os.makedirs(os.path.join(output_train_dir, "gts"), exist_ok=True)
#             shutil.copy(gt_file, os.path.join(os.path.join(output_train_dir, "gts"), file[7:]))
#
# for cls in tqdm(os.listdir(source_svd_val_dir)):
#     cls_dir = os.path.join(source_svd_val_dir, cls)
#     if not os.path.isdir(cls_dir):
#         continue
#     for file in os.listdir(cls_dir):
#         if file.endswith('png') and file.startswith('output_'):
#             os.makedirs(os.path.join(output_val_dir, "inputs"), exist_ok=True)
#             shutil.copy(os.path.join(cls_dir, file), os.path.join(os.path.join(output_val_dir, "inputs"), file[7:]))
#             gt_file = os.path.join(source_orig_dir, cls, file[7:]).replace('png', 'JPEG')
#             os.makedirs(os.path.join(output_val_dir, "gts"), exist_ok=True)
#             shutil.copy(gt_file, os.path.join(os.path.join(output_val_dir, "gts"), file[7:]))
                        

# resize the images to 512x512, and save them to a new folder: inputs_512, gts_512
output_dirs = [os.path.join(nullspace_dir, "data/%s"%dataset)] #[output_train_dir, output_val_dir]
for output_dir in output_dirs:
    inputs = r'/home/fujiwara/project/PhoCoLens/SVDeconv/output/diffusercam/fft-diffusercam/train' # os.path.join(output_dir, 'diffuser_images')
    gts = r'/home/fujiwara/project/PhoCoLens/data/diffusercam/ground_truth_lensed' # os.path.join(output_dir, 'ground_truth_lensed')
    
    # create the new folders for resized images
    inputs_512 = os.path.join(output_dir, 'inputs_512')
    gts_512 = os.path.join(output_dir, 'gts_512')
    for dir in [inputs_512, gts_512]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    files = os.listdir(inputs)
    files = [file for file in files if file.endswith('.png')]
    for file in tqdm(files):
        img = cv2.imread(os.path.join(inputs, file))
        img = cv2.resize(img, (512, 512))
        new_file = file.replace('.png', '.jpg').lstrip('output_')
        cv2.imwrite(os.path.join(inputs_512, new_file), img)

    files = os.listdir(gts)
    files = [file for file in files if file.endswith('.npy')]
    for file in tqdm(files):
        img = np.load(os.path.join(gts, file))
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.resize(img, (512, 512))
        img = cv2.flip(img, 0)
        new_file = file.replace('.npy', '.jpg')
        cv2.imwrite(os.path.join(gts_512, new_file), img)