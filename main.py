import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from tqdm import tqdm
import cv2
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source-video', type=str, required=True, help='要处理视频的绝对路径')
parser.add_argument('-t', '--target-root', type=str, default='./assets', help='要保存到的目录')
parser.add_argument('-d', '--device', type=str, default='cuda:0', help="设备号，默认为'cuda:0'")
parser.add_argument('-m', '--model', type=str, default='model/sam_vit_h_4b8939.pth', help='下载的模型位置')
args = parser.parse_args()

# define color
np.random.seed(0)
colors = [np.random.randint(0, 256, (3,), dtype=np.uint8) for _ in range(100)]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask = np.zeros(shape=(*sorted_anns[0]['segmentation'].shape, 3), dtype=np.uint8)
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        mask[m == 1] = colors[i]
    return mask.astype(np.uint8)


if __name__ == '__main__':
    cap_read = cv2.VideoCapture(args.source_video)
    fps = cap_read.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap_read.isOpened():
        ret, frame = cap_read.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
    cap_read.release()

    device = torch.device(args.device)
    sam = sam_model_registry["default"](checkpoint=args.model)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames[0].shape[: -1]
    save_path = join(args.target_root, "generate.mp4")
    cap_write = cv2.VideoWriter(save_path, fourcc, fps, (width, height), True)
    for frame in tqdm(frames):
        mask = mask_generator.generate(frame)
        mask = show_anns(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        cap_write.write(mask)
    cap_write.release()
    print(f'video save to {save_path}')
