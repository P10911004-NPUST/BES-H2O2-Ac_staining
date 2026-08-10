import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import skimage as sk
from skimage import io
import numpy as np
import pandas as pd
import czifile

def scale_min_max(arr, range = (0, 1)):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * (range[1] - range[0]) + range[0]
    return arr

def imsave(arr, output_path: str = ".", format = "tiff"):
    dirname = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if isinstance(arr, Image.Image):
        arr.save(output_path, format = format)
    else:
        Image.fromarray(arr).save(output_path, format = format)

date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

img_dir = "C:/jklai/project/rice_heat-stress/experiments/BES-H2O2-Ac/20260810_peptide-recover/czi"
img_list = [i for i in os.listdir(img_dir) if i.endswith((".czi"))]
print(f"\n{len(img_list)} images.\n")

output_folder = os.path.join(os.path.dirname(img_dir), "OUT_" + os.path.basename(img_dir))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data = {
    "img_dirname": [], 
    "img_basename": [], 
    "ntrack": [],
    "Zstacks": [],
    "img_size_pixels": [],
    "img_size_um": [],
    "resolution (um/pixel)": [],
    "BES_area": [], 
    "BES_total_intensity": [], 
    "BES_avg_intensity": [],
    "distance_from_root_tip_pixels": [],
    "note": []
}

for i in img_list:
    print(f"Processing {os.path.basename(i)}")
    czi = czifile.CziFile(os.path.join(img_dir, i))
    Metadata = czi.metadata(raw=False)["ImageDocument"]["Metadata"]
    ImageScaling = Metadata["ImageScaling"]
    ImagePixelSize, _ = ImageScaling["ImagePixelSize"]
    Magnification = ImageScaling["ScalingComponent"][1]["Magnification"]  # MTBObjectiveChanger
    microns_per_pixel = ImagePixelSize / Magnification
    pixels_per_micron = Magnification / ImagePixelSize
    czi_arr = czi.asarray()
    _, _, track, zstack, width, height, _ = czi_arr.shape

    BES = np.max(czi_arr[0, 0, 0, :, :, :, 0], axis=0)
    T_PMT = np.min(czi_arr[0, 0, 1, :, :, :, 0], axis=0)

    total_intensity = BES.sum()
    staining_area = np.sum(BES > 0)
    avg_intensity = total_intensity / staining_area
    data["img_dirname"].append(img_dir)
    data["img_basename"].append(i)
    data["ntrack"].append(track)
    data["Zstacks"].append(zstack)
    data["img_size_pixels"].append(f'{height} x {width}')
    data["img_size_um"].append(f'{height * microns_per_pixel} x {width * microns_per_pixel}')
    data["resolution (um/pixel)"].append(microns_per_pixel)
    data["BES_area"].append(staining_area)
    data["BES_total_intensity"].append(total_intensity)
    data["BES_avg_intensity"].append(avg_intensity)
    data["distance_from_root_tip_pixels"].append("")
    data["note"].append("")    
    df0 = pd.DataFrame.from_dict(data)
    df0.to_csv(os.path.join(output_folder, f"OUT_BES_{date_time}.csv"), index=False)

    if BES.dtype != np.uint8:
        BES = np.uint8(scale_min_max(BES))

    BES_binary = BES
    #footprint = sk.morphology.disk(3)
    #for i in range(0, 5):
    #    BES_binary = sk.filters.rank.median(np.uint8(BES_binary), footprint)

    threshold = sk.filters.threshold_otsu(BES_binary)
    BES_binary = (BES_binary > threshold)

    gray_arr = np.float32(T_PMT)
    green_norm = np.float32(BES_binary)

    r = gray_arr * (1 - green_norm)
    g = gray_arr * (1 - green_norm) + 255 * green_norm
    b = gray_arr * (1 - green_norm)

    rgb_arr = np.stack([r, g, b], axis=-1)
    rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)

    RGB = Image.fromarray(rgb_arr, mode="RGB")

    imsave(BES, os.path.join(output_folder, "BES", f"{i.removesuffix('.czi')}.tiff"), "TIFF")
    imsave(T_PMT, os.path.join(output_folder, "T_PMT", f"{i.removesuffix('.czi')}.tiff"), "TIFF")
    imsave(RGB, os.path.join(output_folder, "pseudo", f"{i.removesuffix('.czi')}.tiff"), "TIFF")
