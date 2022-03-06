import argparse
import multiprocessing
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import ttach as tta
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from model import MultiSampleDropout

def load_model(saved_model, idx, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )
    model.classifier = MultiSampleDropout(num_classes=num_classes).to(device=device)
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best' + str(idx) + '.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    
    # test time augmentation model wrapper
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.8, 0.9, 1, 1.1]),
        ])

    answer_preds = []
    for fold_idx in range(5):
        model = load_model(model_dir, fold_idx, num_classes, device)
        model = tta.ClassificationTTAWrapper(model, transforms)
        model.to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print(f"{fold_idx} Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                # pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        answer_preds.append(preds)

    answer_preds = np.mean(answer_preds, axis=0)
    answer_value = np.argmax(answer_preds, axis=-1)

    info['ans'] = answer_value
    save_path = os.path.join(output_dir, f'output2.csv') # 올려줄것
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(380, 380), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='EfficientModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './kfoldmodel/exp2'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './kfoldoutput'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
