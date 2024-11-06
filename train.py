import torch.optim as optim
from dataloader import get_loader
from utils import *
import os
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from distortion import *
import time
from network import SC_TDA

# batch size & train\test left\right !!!

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# datatset: cifar10(train/tets) / DIV2K(train) & kodak, CLIC21(test)
parser = argparse.ArgumentParser(description='SC')
parser.add_argument('--training', action='store_false',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='SC_TDA',
                    choices=['SC_TDA'],
                    help='1 models')
parser.add_argument('--channel-type', type=str, default='rayleigh',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=48,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='3',
                    help='snr(fixed:1, 4, 7, 10, 13)')
parser.add_argument('--tda_dim', type=int, default=28,
                    help='dimension(fixed:28, 56, 112)')
parser.add_argument('--model_path', type=str,
                    default='/home/nf/nf/SC_TDA_Sim32_HARQ/4_cifar10_rayleigh_snr3_c48_31152.model',
                    help='model path')
args = parser.parse_args()

class config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:1")
    norm = False   # CIFAR10 DATASET processing(normalize y/n)
    # logger
    print_step = 100
    plot_step = 10000
    model_path = args.model_path
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 50

    # dataset & encoder, decoder parameters
    # CIFAR10
    if args.trainset == 'CIFAR10':
        save_model_freq = 100
        image_dims = (3, 32, 32)
        train_data_dir = "./Dataset/CIFAR10/"
        test_data_dir = "./Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
            # model="TDA_SC", tda_dim=28
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'MNIST':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "./Dataset/MNIST/"
        test_data_dir = "./Dataset/MNIST/"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True, tda_dim=args.tda_dim,
            # model="TDA_SC", tda_dim=28
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True, tda_dim=args.tda_dim,
        )
    # DIV2K
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        train_data_dir = ["./Dataset/HR_Image_dataset/"]
        if args.testset == 'kodak':
            test_data_dir = ["./Dataset/kodak_test/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["./Dataset/CLIC21/"]
        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True, tda_dim=args.tda_dim,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True, tda_dim=args.tda_dim,
        )

# SSIM
if args.trainset == 'CIFAR10':
     CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
     CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

# pretrained model
def load_weights(model_path):
    net_dict = net.state_dict()
    pretrained = torch.load(model_path)
    state_dict = intersect_dicts(pretrained, net_dict, exclude='')
    print(state_dict.keys())
    net_dict.update(state_dict)
    net.load_state_dict(net_dict, strict=True)
    del pretrained


def train_one_epoch(args):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            input_tda = np.load('cifar10_train_tda_norm.npy')
            input_tda = torch.tensor(input_tda).cuda()
            left = 128 * batch_idx
            right = min(50000, 128 * (batch_idx + 1))
            input_tda = input_tda[left:right, :]
            recon_image, image_noisy_tda, CBR, SNR, mse, loss_G, corr = net(input, input_tda)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                    f'ACK {corr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    elif args.trainset == 'MNIST':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            row = torch.zeros([len(input), 1, 28, 2]).cuda()
            col = torch.zeros([len(input), 1, 2, 32]).cuda()
            input = torch.cat([row, input, row], dim=3)
            input = torch.cat([col, input, col], dim=2)
            input = input.repeat(1, 3, 1, 1)
            input_tda = np.load('mnist_train_tda_476.npy')
            input_tda = torch.tensor(input_tda).cuda()
            left = 128 * batch_idx
            right = min(60000, 128 * (batch_idx + 1))
            input_tda = input_tda[left:right, :]
            recon_image, image_noisy_tda, CBR, SNR, mse, loss_G = net(input, input_tda)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                # msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                # msssims.update(msssim)
            else:
                psnrs.update(100)
                # msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    # f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                # msssim = 1 - loss_G
                # msssims.update(msssim)
            else:
                psnrs.update(100)
                # msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    # f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()


def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'MNIST':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    row = torch.zeros([len(input), 1, 28, 2]).cuda()
                    col = torch.zeros([len(input), 1, 2, 32]).cuda()
                    input = torch.cat([row, input, row], dim=3)
                    input = torch.cat([col, input, col], dim=2)
                    input_tda = np.load('cifar10_test_tda_norm.npy')
                    input_tda = torch.tensor(input_tda).cuda()
                    left = 1024 * batch_idx
                    right = min(10000, 1024 * (batch_idx + 1))
                    input_tda = input_tda[left:right, :]
                    recon_image, image_nosiy_tda, CBR, SNR, mse, loss_G, corr = net(input, input_tda, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                        f'ACK {corr}',
                    ]))
                    logger.info(log)
            elif args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    input_tda = np.load('cifar10_test_tda_norm.npy')
                    input_tda = torch.tensor(input_tda).cuda()
                    left = 1024 * batch_idx
                    right = min(10000, 1024 * (batch_idx + 1))
                    input_tda = input_tda[left:right, :]
                    recon_image, image_nosiy_tda, CBR, SNR, mse, loss_G = net(input, input_tda, SNR)

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                        f'ACK {corr}',
                    ]))
                    logger.info(log)
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        # msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        # msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        # msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        # f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        # results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    # print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")


if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SC_TDA(args, config)
    model_path = args.model_path
    load_weights(model_path)
    net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            if (epoch + 1) % config.save_model_freq == 0:
                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
                test()
    else:
        test()

