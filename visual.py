from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import imageio
import numpy as np
import sys

from data.load_data import LoadVisualData
from models.microisp import MicroISPNet
from utils.utils import process_visual_args


to_image = transforms.Compose([
    transforms.ToPILImage(),
])

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Processing command arguments
restore_epoch, dataset_dir = process_visual_args(sys.argv)

# Dataset size
VISUAL_SIZE = 7

device = torch.device("cuda")
print("CUDA visible devices: " + str(torch.cuda.device_count()))
print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

visual_dataset = LoadVisualData(dataset_dir, VISUAL_SIZE, full_resolution=True)
visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=4,
                           pin_memory=True, drop_last=False)

generator = MicroISPNet(block_num=2).to(device)
generator.load_state_dict(torch.load("checkpoints/microisp" + "_epoch_" + str(restore_epoch) + ".pth"))

generator.eval()
with torch.no_grad():
    visual_iter = iter(visual_loader)
    for j in range(len(visual_loader)):
        torch.cuda.empty_cache()

        raw_image = next(visual_iter)
        raw_image = raw_image.to(device, non_blocking=True)

        enhanced = generator(raw_image.detach())
        enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

        imageio.imwrite("results/microisp_img_" + str(j) + ".jpg", enhanced)