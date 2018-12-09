import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from options import TrainOptions
from wgan_gp import WGAN_GP

def main():
	opt = TrainOptions().parse()

	# writer
	writer = SummaryWriter(os.path.join(opt.log_dir, 'runs'))

	# dataset
	transform = transforms.Compose([
		transforms.Resize((opt.input_size, opt.input_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
	if opt.dataset == 'cifar10':
		dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
	elif opt.dataset == 'cifar100':
		dataset = torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transform)
	elif opt.dataset == 'stl10':
		dataset = torchvision.datasets.STL10('data', split='train', download=True, transform=transform)
	loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

	# model
	wgan_gp = WGAN_GP(opt)
	wgan_gp.train(loader, opt, writer)


if __name__ == '__main__':
	main()