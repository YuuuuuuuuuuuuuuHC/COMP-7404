from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm

import torch
import imageio
import numpy as np
import math
import sys

from data.load_data import LoadData, LoadVisualData
from utils.ssim import SSIM
from models.microisp import MicroISPNet
from models.vgg import vgg_19
from utils.utils import normalize_batch, process_train_args


to_image = transforms.Compose([
    transforms.ToPILImage(),
])

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Processing command arguments
num_train_epochs, batch_size, learning_rate, restore_epoch, dataset_dir = process_train_args(sys.argv)

# Dataset size
TRAIN_SIZE = 16000
EVAL_SIZE = 4000
VISUAL_SIZE = 10

device = torch.device("cuda")
print("CUDA visible devices: " + str(torch.cuda.device_count()))
print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

# Creating dataset loaders
train_dataset = LoadData(dataset_dir, TRAIN_SIZE, eval=False, test=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True)

eval_dataset = LoadData(dataset_dir, EVAL_SIZE, eval=True, test=False)
eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                         pin_memory=True, drop_last=False)

visual_dataset = LoadVisualData(dataset_dir, VISUAL_SIZE, full_resolution=False)
visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=4,
                           pin_memory=True, drop_last=False)

# Creating image processing network and optimizer
generator = MicroISPNet(block_num=2).to(device)
if restore_epoch is not None:
    generator.load_state_dict(torch.load("checkpoints/microisp" + "_epoch_" + str(restore_epoch) + ".pth"))

optimizer = Adam(params=generator.parameters(), lr=learning_rate)

# Losses
MSE_loss = torch.nn.MSELoss()
SSIM_LOSS = SSIM(window_size=11)
VGG_19 = vgg_19(device)

# SummaryWriter
writer = SummaryWriter('logs')

for epoch in range(num_train_epochs):
    torch.cuda.empty_cache()

    train_iter = iter(train_loader)
    for i in tqdm(range(len(train_loader)), desc='Train'):
        optimizer.zero_grad()
        x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        enhanced = generator(x)

        # MSE Loss
        loss_mse = MSE_loss(enhanced, y)

        # SSIM LOSS
        loss_ssim = 1 - SSIM_LOSS(enhanced, y)

        # VGG Loss
        enhanced_vgg = VGG_19(normalize_batch(enhanced))
        target_vgg = VGG_19(normalize_batch(y))
        loss_content = MSE_loss(enhanced_vgg, target_vgg)

        # Total Loss

        # stage1
        total_loss = loss_mse

        # stage2
        # total_loss = loss_content + 0.5 * loss_ssim + 0.5 * loss_mse

        # stage3
        # total_loss = loss_ssim + 0.5 * loss_mse

        total_loss.backward()
        optimizer.step()

    # Save the model that corresponds to the current epoch
    generator.eval().cpu()
    torch.save(generator.state_dict(), "checkpoints/microisp" + "_epoch_" + str(epoch + 1) + ".pth")
    generator.to(device).train()

    # Save visual results for several eval images
    generator.eval()
    with torch.no_grad():
        visual_iter = iter(visual_loader)
        for j in range(len(visual_loader)):
            torch.cuda.empty_cache()

            raw_image = next(visual_iter)
            raw_image = raw_image.to(device, non_blocking=True)

            enhanced = generator(raw_image.detach())
            enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

            imageio.imwrite("results/microisp_img_" + str(j) + "_epoch_" + str(epoch + 1) + ".jpg", enhanced)

    loss_mse_eval = 0
    loss_psnr_eval = 0
    loss_vgg_eval = 0
    loss_ssim_eval = 0

    generator.eval()
    with torch.no_grad():
        eval_iter = iter(eval_loader)
        for j in tqdm(range(len(eval_loader)), desc='Eval'):
            x, y = next(eval_iter)
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

    loss_mse_eval = loss_mse_eval * batch_size / EVAL_SIZE
    loss_psnr_eval = loss_psnr_eval * batch_size / EVAL_SIZE
    loss_vgg_eval = loss_vgg_eval * batch_size / EVAL_SIZE
    loss_ssim_eval = loss_ssim_eval * batch_size / EVAL_SIZE

    writer.add_scalar('add_scalar/psnr', loss_psnr_eval, epoch + 1)
    writer.add_scalar('add_scalar/ssim', loss_ssim_eval, epoch + 1)

    print("Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f, ssim: %.4f" % (epoch + 1, loss_mse_eval, loss_psnr_eval,
                                                                      loss_vgg_eval, loss_ssim_eval))

    generator.train()
