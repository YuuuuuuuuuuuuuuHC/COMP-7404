from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import math
import sys

from data.load_data import LoadData
from utils.ssim import SSIM
from models.microisp import MicroISPNet
from models.vgg import vgg_19
from utils.utils import normalize_batch, process_test_args


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Processing command arguments
batch_size, restore_epoch, dataset_dir = process_test_args(sys.argv)

# Dataset size
TEST_SIZE = 4000

device = torch.device("cuda")
print("CUDA visible devices: " + str(torch.cuda.device_count()))
print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

test_dataset = LoadData(dataset_dir, TEST_SIZE, eval=False, test=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                         pin_memory=True, drop_last=False)

generator = MicroISPNet(block_num=2).to(device)
generator.load_state_dict(torch.load("checkpoints/microisp" + "_epoch_" + str(restore_epoch) + ".pth"))

# Losses
MSE_loss = torch.nn.MSELoss()
SSIM_LOSS = SSIM(window_size=11)
VGG_19 = vgg_19(device)

loss_mse_eval = 0
loss_psnr_eval = 0
loss_vgg_eval = 0
loss_ssim_eval = 0

generator.eval()
with torch.no_grad():
    test_iter = iter(test_loader)
    for j in tqdm(range(len(test_loader)), desc='Test'):
        x, y = next(test_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        enhanced = generator(x)

        loss_mse_temp = MSE_loss(enhanced, y).item()
        loss_mse_eval += loss_mse_temp

        loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

        loss_ssim_eval += SSIM_LOSS(enhanced, y)

        enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
        target_vgg_eval = VGG_19(normalize_batch(y)).detach()
        loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()

loss_mse_eval = loss_mse_eval * batch_size / TEST_SIZE
loss_psnr_eval = loss_psnr_eval * batch_size / TEST_SIZE
loss_vgg_eval = loss_vgg_eval * batch_size / TEST_SIZE
loss_ssim_eval = loss_ssim_eval * batch_size / TEST_SIZE

print("mse: %.4f, psnr: %.4f, vgg: %.4f, ssim: %.4f" % (loss_mse_eval, loss_psnr_eval, loss_vgg_eval, loss_ssim_eval))

