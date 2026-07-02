#!/usr/bin/env python
"""
Run inference on 10 test images and save visualized results.
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.visualization import overlay_bbox_cv


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        self.grayscale = cfg.data.val.get("grayscale", False)
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            if self.grayscale:
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Cant load image: {img}")
                img = img[..., None]
            else:
                img = cv2.imread(img)
                if img is None:
                    raise FileNotFoundError(f"Cant load image: {img}")
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = np.array([height])
        img_info["width"] = np.array([width])
        img_info["id"] = np.array([0])
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        # post_process expects batch format: wrap single sample artifacts in lists
        meta["warp_matrix"] = [meta["warp_matrix"]]
        if meta["img"].ndim == 2:
            img_tensor = torch.from_numpy(meta["img"][None, ...])
        else:
            img_tensor = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        meta["img"] = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results


def draw_results(img, dets, class_names, score_thresh=0.5):
    """Draw predicted bboxes on image."""
    vis_img = img.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    elif vis_img.shape[2] == 1:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    return overlay_bbox_cv(vis_img, dets, class_names, score_thresh)


def main():
    config_path = "config/nanodet-plus-m_480x640_barcode30k.yml"
    model_path = "workspace/nanodet-plus-m_barcode30k/model_last.ckpt"
    test_json = "dataset/barcode30k_final/test/test.json"
    test_img_dir = "dataset/barcode30k_final/test/images"
    output_dir = "workspace/nanodet-plus-m_barcode30k/test_predictions"
    num_samples = 10
    score_thresh = 0.5

    os.makedirs(output_dir, exist_ok=True)

    # Load config
    load_config(cfg, config_path)

    # Create predictor
    logger = Logger(-1, output_dir, False)
    predictor = Predictor(cfg, model_path, logger, device="cuda:0")

    # Load test annotations and sample 10 images
    with open(test_json) as f:
        test_data = json.load(f)

    images = test_data["images"]
    random.seed(42)
    sampled = random.sample(images, min(num_samples, len(images)))

    for img_info in sampled:
        file_name = img_info["file_name"]
        img_path = os.path.join(test_img_dir, file_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        meta, results = predictor.inference(img_path)
        dets = results[0] if 0 in results else {}

        # Read original image for visualization
        if cfg.data.val.get("grayscale", False):
            raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            raw_img = cv2.imread(img_path)

        vis_img = draw_results(raw_img, dets, cfg.class_names, score_thresh)

        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, vis_img)
        print(f"Saved: {output_path}")

    print(f"\nDone. {len(sampled)} visualized images saved to: {output_dir}")


if __name__ == "__main__":
    main()
