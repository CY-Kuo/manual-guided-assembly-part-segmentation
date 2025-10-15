# main.py
import os, sys, json, csv, cv2, yaml, random, numpy as np, pandas as pd, matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
from pycocotools import mask as maskUtils
sys.path.append("/src")
from pathlib import Path
from IKEAVideo.dataloader.dataset_keyframe import KeyframeDataset
from IKEAVideo.dataloader.assembly_video   import load_video, load_frame
from yolo_dataset_builder import build

# ─── CONFIG ──────────────────────────────────────────────────────
annotation = "../data/data.json"
video_dir   = "../data/video"
manual_dir  = "../data/manual_img"
obj_dir     = "../data/parts"
pdf_dir     = "../data/pdfs"
num_of_data = 98

ds = KeyframeDataset(annotation, video_dir,
                     transform=None,
                     load_into_mem=False,
                     verbose=False, debug=False,
                     obj_dir=obj_dir, num_of_data=num_of_data,
                     manual_img_dir=manual_dir,
                     pdf_dir=pdf_dir,
                     demo_print=False, demo_viz=False)

# filter only keyframes
#keyframes = [f for f in ds.data if f.get("is_keyframe")]
#keyframes = [f for f in ds.data]
keyframes = [f for i, f in enumerate(ds.data) if i % 3 == 0]

# 1) YOLO-seg dataset
mini = Path("mini_2")
build(mini, keyframes, video_dir)

print("✅ All datasets built!")
