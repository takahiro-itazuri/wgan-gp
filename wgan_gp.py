import os
import sys
import math

import torch
import torchvision
from torch import nn, optim
from torch.autograd import grad
from torch.nn import functional as F
from torchvision.utils import make_grid

from layers import ConditionalBatchNorm2d, LayerNorm


def NormLayer(type, num_features, num_classes=-1, affine=True):
	if type == 'batchnorm':
		return nn.BatchNorm2d(num_features, affine=affine)
	elif type == 'layernorm':
		return LayerNorm(num_features, affine=affine)
	elif type == 'conditional_batchnorm':
		if num_classes == -1:
			raise ValueError('expected positive value (got -1)')
		return ConditionalBatchNorm2d(num_features, num_classes, affine=affine)


def UpLayer(type, scale_factor=2):
	if type == 'nearest':
		return nn.UpsamplingNearest2d(scale_factor=scale_factor)
	elif type == 'bilinear':
		return nn.UpsamplingBilinear2d(scale_factor=2)


def ActLayer(type, negative_slope=0.2):
	if type == 'relu':
		return nn.ReLU(inplace=True)
	elif type == 'leakyrelu':
		return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)


class Conv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(Conv, self).__init__()
		self.he_init = he_init
		if kernel_size == 1:
			self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
		elif kernel_size == 3:
			self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)

	def forward(self, x):
		return self.conv(x)


class DownConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(DownConv, self).__init__()
		self.conv = nn.Sequential(
			nn.AvgPool2d(kernel_size=2),
			Conv(in_channels, out_channels, kernel_size, bias, he_init)
		)

	def forward(self, x):
		return self.conv(x)


class ConvDown(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(ConvDown, self).__init__()
		self.conv = nn.Sequential(
			Conv(in_channels, out_channels, kernel_size, bias, he_init),
			nn.AvgPool2d(kernel_size=2)
		)

	def forward(self, x):
		return self.conv(x)


class UpConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(UpConv, self).__init__()
		self.conv = nn.Sequential(
			nn.UpsamplingNearest2d(scale_factor=2),
			Conv(in_channels, out_channels, kernel_size, bias, he_init)
		)
	
	def forward(self, x):
		return self.conv(x)


class ResBlock32(nn.Module):
	def __init__(self, in_channels, out_channels, norm_type, act_type, resample=None, num_classes=-1):
		super(ResBlock32, self).__init__()
		if resample not in [None, 'up', 'down']:
			raise ValueError('resample method is None, "up", or "down" (got {})'.format(resample))
		self.num_classes = num_classes
		self.condition = (num_classes != -1)
		self.need_shortcut_conv = (in_channels != out_channels) or (resample is not None)

		# shortcut convolution
		if resample == 'down':
			self.shortcut_conv = ConvDown(in_channels, out_channels, 1, he_init=False)
		elif resample == 'up':
			self.shortcut_conv = UpConv(in_channels, out_channels, 1, he_init=False)
		elif resample == None and self.need_shortcut_conv:
			self.shortcut_conv = Conv(in_channels, out_channels, 1, he_init=False)

		# convolution
		if resample == 'down':
			self.conv1 = Conv(in_channels, in_channels, 3)
			self.conv2 = ConvDown(in_channels, out_channels, 3)
		elif resample == 'up':
			self.conv1 = UpConv(in_channels, out_channels, 3)
			self.conv2 = Conv(out_channels, out_channels, 3)
		elif resample is None:
			self.conv1 = Conv(in_channels, out_channels, 3)
			self.conv2 = Conv(out_channels, out_channels, 3)
		
		# activation
		self.act1 = ActLayer(act_type)
		self.act2 = ActLayer(act_type)

		# normalization
		self.norm1 = NormLayer(norm_type, in_channels, num_classes)
		self.norm2 = NormLayer(norm_type, out_channels, num_classes)

	def forward(self, x, t=None):
		o = self.norm1(x, t) if self.condition else self.norm1(x)
		o = self.act1(o)
		o = self.conv1(o)
		o = self.norm2(o, t) if self.condition else self.norm2(o)
		o = self.act2(o)
		o = self.conv2(o)
		
		return o + self.shortcut(x)

	def shortcut(self, x):
		if self.need_shortcut_conv:
			return self.shortcut_conv(x)
		else:
			return x


class OptimizedBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm_type, act_type):
		super(OptimizedBlock, self).__init__()
		self.conv = nn.Sequential(
			Conv(in_channels, out_channels, 3),
			ActLayer(act_type),
			ConvDown(out_channels, out_channels, 3)
		)
		self.shortcut = DownConv(in_channels, out_channels, 1, he_init=False)
	
	def forward(self, x):
		return self.conv(x) + self.shortcut(x)


class Generator32(nn.Module):
	def __init__(self, nz, nc, ngf, norm_type='batchnorm', act_type='leakyrelu', num_classes=-1):
		super(Generator32, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.num_classes = num_classes
		self.condition = (num_classes != -1)

		self.fc1 = nn.Linear(nz, 4 * 4 * ngf)
		self.block2 = ResBlock32(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.block3 = ResBlock32(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.block4 = ResBlock32(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.final = nn.Sequential(
			NormLayer('batchnorm', ngf),
			ActLayer(act_type),
			Conv(ngf, nc, 3, he_init=False),
			nn.Tanh()
		)

	def forward(self, z, t=None):
		x = self.fc1(z)
		x = x.view(-1, self.ngf, 4, 4)
		x = self.block2(x, t)
		x = self.block3(x, t)
		x = self.block4(x, t)
		x = self.final(x)
		return x


class Critic32(nn.Module):
	def __init__(self, nz, nc, ndf, norm_type='instancenorm', act_type='leakyrelu', num_classes=-1):
		super(Critic32, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ndf = ndf
		self.num_classes = num_classes
		self.condition = (num_classes != -1)
		
		self.block1 = OptimizedBlock(nc, ndf, norm_type, act_type)
		self.block2 = ResBlock32(ndf, ndf, norm_type, act_type, 'down')
		self.block3 = ResBlock32(ndf, ndf, norm_type, act_type, None)
		self.block4 = ResBlock32(ndf, ndf, norm_type, act_type, None)
		self.pool = nn.Sequential(
			ActLayer(act_type),
			nn.AvgPool2d(kernel_size=8)
		)
		self.wgan = nn.Linear(ndf, 1)
		if self.condition:
			self.ac = nn.Linear(ndf, num_classes)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.pool(x).view(x.size(0), -1)

		if self.condition:
			return self.wgan(x), self.ac(x)
		else:
			return self.wgan(x)


class ResBlock64(nn.Module):
	def __init__(self, in_channels, out_channels, norm_type, act_type, resample=None, num_classes=-1):
		super(ResBlock64, self).__init__()
		if resample not in [None, 'up', 'down']:
			raise ValueError('resample method is None, "up", or "down" (got {})'.format(resample))
		self.num_classes = num_classes
		self.condition = (num_classes != -1)
		self.need_shortcut_conv = (in_channels != out_channels) or (resample is not None)

		# shortcut convolution
		if resample == 'down':
			self.shortcut_conv = DownConv(in_channels, out_channels, 1, he_init=False)
		elif resample == 'up':
			self.shortcut_conv = UpConv(in_channels, out_channels, 1, he_init=False)
		elif resample == None and self.need_shortcut_conv:
			self.shortcut_conv = Conv(in_channels, out_channels, 1, he_init=False)

		# convolution
		if resample == 'down':
			self.conv1 = Conv(in_channels, in_channels, 3)
			self.conv2 = ConvDown(in_channels, out_channels, 3)
		elif resample == 'up':
			self.conv1 = UpConv(in_channels, out_channels, 3)
			self.conv2 = Conv(out_channels, out_channels, 3)
		elif resample is None:
			self.conv1 = Conv(in_channels, in_channels, 3)
			self.conv2 = Conv(in_channels, out_channels, 3)
		
		# activation
		self.act1 = ActLayer(act_type)
		self.act2 = ActLayer(act_type)

		# normalization
		self.norm1 = NormLayer(norm_type, in_channels, num_classes)
		if resample == 'down' or resample == None:
			self.norm2 = NormLayer(norm_type, in_channels, num_classes)
		elif resample == 'up':
			self.norm2 = NormLayer(norm_type, out_channels, num_classes)

	def forward(self, x, t=None):
		o = self.norm1(x, t) if self.condition else self.norm1(x)
		o = self.act1(o)
		o = self.conv1(o)
		o = self.norm2(o, t) if self.condition else self.norm2(o)
		o = self.act2(o)
		o = self.conv2(o)
		
		return o + self.shortcut(x)

	def shortcut(self, x):
		if self.need_shortcut_conv:
			return self.shortcut_conv(x)
		else:
			return x


class Generator64(nn.Module):
	def __init__(self, nz, nc, ngf, norm_type='batchnorm', act_type='relu', num_classes=-1):
		super(Generator64, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.num_classes = num_classes
		self.condition = (num_classes != -1)

		self.fc1 = nn.Linear(nz, 4 * 4 * 8 * ngf)
		self.block2 = ResBlock64(8 * ngf, 8 * ngf, norm_type, act_type, 'up', num_classes)
		self.block3 = ResBlock64(8 * ngf, 4 * ngf, norm_type, act_type, 'up', num_classes)
		self.block4 = ResBlock64(4 * ngf, 2 * ngf, norm_type, act_type, 'up', num_classes)
		self.block5 = ResBlock64(2 * ngf, 1 * ngf, norm_type, act_type, 'up', num_classes)
		self.final = nn.Sequential(
			NormLayer('batchnorm', ngf),
			ActLayer(act_type),
			Conv(1 * ngf, nc, 3),
			nn.Tanh()
		)

	def forward(self, z, t=None):
		x = self.fc1(z)
		x = x.view(-1, 8 * self.ngf, 4, 4)
		x = self.block2(x, t)
		x = self.block3(x, t)
		x = self.block4(x, t)
		x = self.block5(x, t)
		x = self.final(x)
		return x


class Critic64(nn.Module):
	def __init__(self, nz, nc, ndf, norm_type='instancenorm', act_type='relu', num_classes=-1):
		super(Critic64, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ndf = ndf
		self.num_classes = num_classes
		self.condition = (num_classes != -1)
		
		
		self.first = Conv(nc, ndf, 3, he_init=False)
		self.block1 = ResBlock64(1 * ndf, 2 * ndf, norm_type, act_type, 'down')
		self.block2 = ResBlock64(2 * ndf, 4 * ndf, norm_type, act_type, 'down')
		self.block3 = ResBlock64(4 * ndf, 8 * ndf, norm_type, act_type, 'down')
		self.block4 = ResBlock64(8 * ndf, 8 * ndf, norm_type, act_type, 'down')
		self.wgan = nn.Linear(4 * 4 * 8 * ndf, 1)
		if self.condition:
			self.ac = nn.Linear(4 * 4 * 8 * ndf, num_classes)

	def forward(self, x):
		x = self.first(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = x.view(x.size(0), -1)

		if self.condition:
			return self.wgan(x), self.ac(x)
		else:
			return self.wgan(x)


def initialize_weights(model):
	for m in model.modules():
		if isinstance(m, Conv):
			if m.he_init:
				nn.init.kaiming_uniform_(m.conv.weight)
			else:
				nn.init.xavier_uniform_(m.conv.weight)
			if m.conv.bias is not None:
				nn.init.constant_(m.conv.bias, 0.0)
		elif isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0.0)


class WGAN_GP(object):
	def __init__(self, opt):
		# hyperparameters
		self.num_critic = opt.num_critic
		self.lambda_gp = opt.lambda_gp
		self.nz = 128
		self.nc = 3
		self.ngf = 128
		self.ndf = 128
		self.input_size = opt.input_size
		self.num_classes = opt.num_classes
		self.condition = opt.condition
		if self.condition:
			self.num_samples = self.num_classes * self.num_classes
		else:
			self.num_samples = 64

		self.device = opt.device

		# model
		if self.condition:
			if self.input_size == 32:
				self.G = Generator32(self.nz, self.nc, self.ngf, opt.G_norm_type, opt.G_act_type, self.num_classes)
				self.C = Critic32(self.nz, self.nc, self.ndf, opt.C_norm_type, opt.C_act_type, self.num_classes)
			elif self.input_size == 64:
				self.G = Generator64(self.nz, self.nc, self.ngf, opt.G_norm_type, opt.G_act_type, self.num_classes)
				self.C = Critic64(self.nz, self.nc, self.ndf, opt.C_norm_type, opt.C_act_type, self.num_classes)
		else:
			if self.input_size == 32:
				self.G = Generator32(self.nz, self.nc, self.ngf, opt.G_norm_type, opt.G_act_type)
				self.C = Critic32(self.nz, self.nc, self.ndf, opt.C_norm_type, opt.C_act_type)
			elif self.input_size == 64:
				self.G = Generator64(self.nz, self.nc, self.ngf, opt.G_norm_type, opt.G_act_type)
				self.C = Critic64(self.nz, self.nc, self.ndf, opt.C_norm_type, opt.C_act_type)

		if torch.cuda.device_count() > 1:
			print('multiple GPU mode!')
			self.G = nn.DataParallel(self.G)
			self.C = nn.DataParallel(self.C)
		
		self.G = self.G.to(self.device)
		self.C = self.C.to(self.device)

		if opt.train:
			initialize_weights(self.G)
			initialize_weights(self.C)
			self.G_optimizer = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
			self.C_optimizer = optim.Adam(self.C.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

		# criterion
		if self.condition:
			self.ac_loss = nn.CrossEntropyLoss()

		# fixed input
		if self.condition:
			self.sample_z = torch.zeros((self.num_samples, self.nz)).to(self.device)
			for i in range(self.num_classes):
				self.sample_z[i * self.num_classes] = torch.randn((1, self.nz))
				for j in range(1, self.num_classes):
					self.sample_z[i * self.num_classes + j] = self.sample_z[i * self.num_classes]
			self.sample_z = self.sample_z.to(self.device)

			temp_t = torch.tensor(range(self.num_classes))
			self.sample_t = temp_t.repeat(self.num_classes)
			self.sample_t = self.sample_t.to(self.device)
		else:
			self.sample_z = torch.randn((self.num_samples, self.nz)).to(self.device)

	def train(self, loader, opt, writer=None):		
		itr = 0
		running_EMD = 0.0
		if self.condition:
			running_AC_loss = 0.0

		while True:
			for i, (x_real, t) in enumerate(loader):
				if x_real.size(0) != opt.batch_size:
					break

				x_real = x_real.to(self.device)
				z = torch.randn((opt.batch_size, self.nz)).to(self.device)
				if self.condition:
					t = t.to(self.device)
					# t_onehot = torch.zeros((opt.batch_size, self.num_classes)).scatter_(1, t.type(torch.LongTensor).unsqueeze(1), 1).to(self.device)

				# === update D network === #
				self.C.train()
				self.G.train()
				self.C_optimizer.zero_grad()
				if self.condition:
					C_real, t_real = self.C(x_real)
					C_real = torch.mean(C_real)
					AC_real_loss = self.ac_loss(t_real, t)

					x_fake = self.G(z, t).detach()
					C_fake, t_fake = self.C(x_fake)
					C_fake = torch.mean(C_fake)
					AC_fake_loss = self.ac_loss(t_fake, t)

					AC_loss = AC_real_loss + AC_fake_loss
				else:
					C_real = self.C(x_real)
					C_real = torch.mean(C_real)

					x_fake = self.G(z).detach()
					C_fake = self.C(x_fake)
					C_fake = torch.mean(C_fake)
				# EMD
				EMD = C_real - C_fake

				# gradient penalty
				alpha = torch.rand((opt.batch_size, 1, 1, 1)).to(self.device)
				x_hat = alpha * x_real.data + (1 - alpha) * x_fake.data
				x_hat.requires_grad = True

				if self.condition:
					C_hat, _ = self.C(x_hat)
				else:
					C_hat = self.C(x_hat)
				gradients = grad(outputs=C_hat, inputs=x_hat, grad_outputs=torch.ones(C_hat.size()).to(self.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
				GP = self.lambda_gp * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

				if self.condition:
					C_loss = -EMD + GP + AC_loss
				else:
					C_loss = -EMD + GP
				C_loss.backward()
				self.C_optimizer.step()

				if ((i + 1) % self.num_critic) == 0:
					itr += 1

					# === update G === #
					self.G.train()
					self.G_optimizer.zero_grad()

					if self.condition:
						x_fake = self.G(z, t)
						C_fake, t_fake = self.C(x_fake)
						C_fake = torch.mean(C_fake)
						AC_fake_loss = self.ac_loss(t_fake, t)
						G_loss = -C_fake + AC_fake_loss
					else:
						x_fake = self.G(z)
						C_fake = self.C(x_fake)
						C_fake = torch.mean(x_fake)
						G_loss = -C_fake

					G_loss.backward()
					self.G_optimizer.step()

					# log
					if writer is not None:
						writer.add_scalars('Loss', {'EMD': EMD.item()}, global_step=itr)
						writer.add_scalars('Others', {'C_real': C_real.item(), 'C_fake': C_fake.item(), 'GP': GP.item()}, global_step=itr)
						if self.condition:
							writer.add_scalars('Loss', {'AC_loss': AC_loss.item()}, global_step=itr)
							writer.add_scalars('Others', {'AC_real_loss': AC_real_loss.item(), 'AC_fake_loss': AC_fake_loss.item()}, global_step=itr)
						samples = self.generate()
						writer.add_image('generated samples', make_grid(samples, nrow=int(math.sqrt(self.num_samples))), global_step=itr)

					# standard output
					running_EMD += EMD.item()
					if self.condition:
						running_AC_loss += AC_loss.item()
					if itr % 100 == 0:
						commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}'.format(itr, running_EMD / 100)
						running_EMD = 0.0
						if self.condition:
							commandline_output += ', AC_loss: {:.4f}'.format(running_AC_loss / 100)
							running_AC_loss = 0.0
						commandline_output += '\n'
					else:
						commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}'.format(itr, EMD.item())
						if self.condition:
							commandline_output += ', AC_loss: {:.4f}'.format(AC_loss.item())
					sys.stdout.write(commandline_output)
					sys.stdout.flush()

				# save model
				if itr % opt.checkpoint == 0:
					G_path = os.path.join(opt.log_dir, 'G_{:d}itr.pkl'.format(itr))
					C_path = os.path.join(opt.log_dir, 'C_{:d}itr.pkl'.format(itr))
					self.save(G_path, C_path)

				if itr == opt.num_itrs:
					break
			
			if itr == opt.num_itrs:
				break

	def generate(self, fix=True):
		self.G.eval()

		if fix:
			if self.condition:
				samples = self.G(self.sample_z, self.sample_t)
			else:
				samples = self.G(self.sample_z)
		else:
			if self.condition:
				samples = self.G(torch.randn((self.num_samples, self.nz)).to(self.device), self.sample_t)
			else:
				samples = self.G(torch.randn((self.num_samples, self.nz)).to(self.device))

		return (samples + 1.) / 2.

	def save(self, G_path, C_path):
		torch.save(self.G.state_dict(), G_path)
		torch.save(self.C.state_dict(), C_path)

	def load(self, G_path, C_path):
		self.G.load_state_dict(torch.load(G_path))
		self.C.load_state_dict(torch.load(C_path))
