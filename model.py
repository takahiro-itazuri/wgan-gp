import os
import sys

import torch
from torch import nn, optim
from torch.autograd import grad

def initialize_weights(model):
	for m in model.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight.data)
			if m.bias is not None:
				nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.uniform_(m.weight.data)
			nn.init.zeros_(m.bias.data)


def adjust_lr(optimizer, initial_lr, final_lr, itr, num_itrs):
	lr = initial_lr - float(itr) / float(num_itrs) * (initial_lr - final_lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-5, affine=True):
		super(LayerNorm, self).__init__()
		self.eps = eps
		self.affine = affine

		if self.affine:
			self.gamma = nn.Parameter(torch.ones(num_features))
			self.beta = nn.Parameter(torch.zeros(num_features))
	
	def forward(self, x):
		shape = [-1] + [1] * (x.dim() - 1)
		mean = x.view(x.size(0), -1).mean(1).view(*shape)
		std = x.view(x.size(0), -1).std(1).view(*shape)
		x = (x - mean) / (std + self.eps)
		
		if self.affine:
			shape = [1, -1] + [1] * (x.dim() - 2)
			x = self.gamma.view(*shape) * x + self.beta.view(*shape)

		return x


def Conv1x1(in_channels, out_channels, bias=False):
	return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


def Conv3x3(in_channels, out_channels, bias=False):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


def NormLayer(type, num_features, affine=True):
	if type == 'batchnorm':
		return nn.BatchNorm2d(num_features, affine=affine)
	elif type == 'instancenorm':
		return nn.InstanceNorm2d(num_features, affine=affine)
	elif type == 'layernorm':
		return LayerNorm(num_features, affine=affine)


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


class DownConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super(DownConv, self).__init__()
		self.conv = nn.Sequential(
			nn.AvgPool2d(kernel_size=2),
			Conv3x3(in_channels, out_channels) if (kernel_size == 3) else Conv1x1(in_channels, out_channels)
		)

	def forward(self, x):
		return self.conv(x)


class ConvDown(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super(ConvDown, self).__init__()
		self.conv = nn.Sequential(
			Conv3x3(in_channels, out_channels) if (kernel_size == 3) else Conv1x1(in_channels, out_channels),
			nn.AvgPool2d(kernel_size=2)
		)

	def forward(self, x):
		return self.conv(x)


class UpConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super(UpConv, self).__init__()
		self.conv = nn.Sequential(
			nn.UpsamplingNearest2d(scale_factor=2),
			Conv3x3(in_channels, out_channels) if (kernel_size == 3) else Conv1x1(in_channels, out_channels)
		)
	
	def forward(self, x):
		return self.conv(x)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm_type, act_type, resample=None):
		super(ResidualBlock, self).__init__()
		self.need_shortcut_conv = (in_channels != out_channels) or (resample is not None)

		if resample == 'down':
			self.conv = nn.Sequential(
				NormLayer(norm_type, in_channels),
				ActLayer(act_type),
				Conv3x3(in_channels, out_channels),
				NormLayer(norm_type, out_channels),
				ActLayer(act_type),
				ConvDown(out_channels, out_channels)
			)
			self.shortcut_conv = ConvDown(in_channels, out_channels, kernel_size=1)
		elif resample == 'up':
			self.conv = nn.Sequential(
				NormLayer(norm_type, in_channels),
				ActLayer(act_type),
				UpConv(in_channels, out_channels),
				NormLayer(norm_type, out_channels),
				ActLayer(act_type),
				Conv3x3(out_channels, out_channels)
			)
			self.shortcut_conv = UpConv(in_channels, out_channels, kernel_size=1)
		elif resample is None:
			self.conv = nn.Sequential(
				NormLayer(norm_type, in_channels),
				ActLayer(act_type),
				Conv3x3(in_channels, out_channels),
				NormLayer(norm_type, out_channels),
				ActLayer(act_type),
				Conv3x3(out_channels, out_channels)
			)
			if self.need_shortcut_conv:
				self.shortcut_conv = Conv1x1(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		if self.need_shortcut_conv:
			shortcut = self.shortcut_conv(x)
		else:
			shortcut = x

		x = self.conv(x)
		x += shortcut
		return x


class OptimizedBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm_type, act_type):
		super(OptimizedBlock, self).__init__()
		self.conv = nn.Sequential(
			Conv3x3(in_channels, out_channels),
			NormLayer(norm_type, out_channels),
			ActLayer(act_type),
			ConvDown(out_channels, out_channels)
		)
		self.shortcut = DownConv(in_channels, out_channels, kernel_size=1)
	
	def forward(self, x):
		shortcut = self.shortcut(x)
		x = self.conv(x)
		x += shortcut
		return x


class Generator(nn.Module):
	def __init__(self, nz=128, nc=3, ngf=128, norm_type='batchnorm', act_type='leakyrelu'):
		super(Generator, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf

		self.fc1 = nn.Linear(nz, 4 * 4 * ngf, bias=False)
		self.block2 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up')
		self.block3 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up')
		self.block4 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up')
		self.block5 = nn.Sequential(
			NormLayer(norm_type, ngf),
			ActLayer(act_type),
			Conv3x3(ngf, nc),
			nn.Tanh()
		)

	def forward(self, z):
		x = self.fc1(z)
		x = x.view(-1, self.ngf, 4, 4)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		return x


class Discriminator(nn.Module):
	def __init__(self, nz=128, nc=3, ndf=128, norm_type='instancenorm', act_type='leakyrelu'):
		super(Discriminator, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ndf = ndf
		
		self.block1 = OptimizedBlock(nc, ndf, norm_type, act_type)
		self.block2 = ResidualBlock(ndf, ndf, norm_type, act_type, 'down')
		self.block3 = ResidualBlock(ndf, ndf, norm_type, act_type, None)
		self.block4 = ResidualBlock(ndf, ndf, norm_type, act_type, None)
		self.pool5 = nn.Sequential(
			NormLayer(norm_type, ndf),
			ActLayer(act_type),
			nn.AvgPool2d(kernel_size=8)
		)
		self.fc6 = nn.Linear(ndf, 1)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.pool5(x).view(x.size(0), -1)
		x = self.fc6(x)
		return x


class WGAN_GP(object):
	def __init__(self, opt):
		# hyperparameters
		self.num_samples = 64
		self.num_critic = opt.num_critic
		self.lambda_gp = opt.lambda_gp
		self.nz = 128
		self.nc = 3
		self.ngf = 128
		self.ndf = 128

		self.device = opt.device

		# model
		self.G = Generator(self.nz, self.nc, self.ngf, opt.G_norm_type, opt.G_act_type).to(self.device)
		self.D = Discriminator(self.nz, self.nc, self.ndf, opt.D_norm_type, opt.D_act_type).to(self.device)
		if opt.train:
			initialize_weights(self.G)
			initialize_weights(self.D)
			self.G_optimizer = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
			self.D_optimizer = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

		self.sample_z = torch.randn((self.num_samples, self.nz)).to(self.device)

	def train(self, loader, opt, writer=None):
		self.y_real = torch.ones(opt.batch_size, 1).to(self.device)
		self.y_fake = torch.zeros(opt.batch_size, 1).to(self.device)
		
		nitrs = 0
		running_D_real_loss = 0.0
		running_D_fake_loss = 0.0
		running_G_loss = 0.0
		running_EMD = 0.0
		running_GP = 0.0

		while True:
			for itr, (x_real, _) in enumerate(loader):
				if x_real.size(0) != opt.batch_size:
					break

				x_real = x_real.to(opt.device)
				z = torch.randn((opt.batch_size, self.nz)).to(self.device)
				
				# === update D network === #
				self.D.train()
				self.G.train()
				self.D_optimizer.zero_grad()

				# real sample
				D_real = self.D(x_real)
				D_real_loss = torch.mean(D_real)

				# fake sample
				x_fake = self.G(z).detach()
				D_fake = self.D(x_fake)
				D_fake_loss = torch.mean(D_fake)

				# EMD
				EMD = D_real_loss - D_fake_loss

				# gradient penalty
				alpha = torch.rand((opt.batch_size, 1, 1, 1)).to(self.device)
				x_hat = alpha * x_real.data + (1 - alpha) * x_fake.data
				x_hat.requires_grad = True

				D_hat = self.D(x_hat)
				gradients = grad(outputs=D_hat, inputs=x_hat, grad_outputs=torch.ones(D_hat.size()).to(self.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
				GP = self.lambda_gp * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

				D_loss = -EMD + GP
				D_loss.backward(retain_graph=True)
				self.D_optimizer.step()

				if ((itr + 1) % self.num_critic) == 0:
					nitrs += 1

					# === update G === #
					self.G.train()
					self.G_optimizer.zero_grad()

					x_fake = self.G(z)
					D_fake = self.D(x_fake)
					
					G_loss = -torch.mean(D_fake)
					G_loss.backward()
					self.G_optimizer.step()

					# # adjust learning rate
					# adjust_lr(self.D_optimizer, opt.lr, opt.lr / 10, nitrs, opt.num_itrs)
					# adjust_lr(self.G_optimizer, opt.lr, opt.lr / 10,  nitrs, opt.num_itrs)

					# log
					if writer is not None:
						writer.add_scalars('Loss', {'EMD': EMD.item()}, global_step=nitrs)
						writer.add_scalars('Others', {'D_real_loss': D_real_loss.item(), 'D_fake_loss': D_fake_loss.item(), 'G_loss': G_loss.item(), 'GP': GP.item()}, global_step=nitrs)
						samples = self.generate()
						writer.add_image('generated samples', samples, global_step=nitrs)

					# standard output
					running_EMD += EMD.item()
					running_D_real_loss += D_real_loss.item()
					running_D_fake_loss += D_fake_loss.item()
					running_G_loss += G_loss.item()
					running_GP += GP.item()
					if nitrs % 100 == 0:
						commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}, D_real_loss: {:.4f}, D_fake_loss: {:.4f}, G_loss: {:.4f}, GP: {:.4f}\n'.format(nitrs, running_EMD / 100, running_D_real_loss / 100, running_D_fake_loss / 100, running_G_loss / 100, running_GP / 100)
						running_EMD = 0.0
						running_D_real_loss = 0.0
						running_D_fake_loss = 0.0
						running_G_loss = 0.0
						running_GP = 0.0
					else:
						commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}, D_real_loss: {:.4f}, D_fake_loss: {:.4f}, G_loss: {:.4f}, GP: {:.4f}'.format(nitrs, EMD.item(), D_real_loss.item(), D_fake_loss.item(), G_loss.item(), GP.item())
					sys.stdout.write(commandline_output)
					sys.stdout.flush()

				# save model
				if nitrs != 0 and nitrs % opt.checkpoint == 0:
					G_path = os.path.join(opt.log_dir, 'G_{:d}itr.pkl'.format(nitrs))
					D_path = os.path.join(opt.log_dir, 'D_{:d}itr.pkl'.format(nitrs))
					self.save(G_path, D_path)

				if nitrs == opt.num_itrs:
					break
			
			if nitrs == opt.num_itrs:
				break

	def generate(self, fix=True):
		self.G.eval()

		if fix:
			samples = self.G(self.sample_z)
		else:
			samples = self.G(torch.rand((self.num_samples, self.nz)).to(self.device))

		return (samples + 1.) / 2.

	def save(self, G_path, D_path):
		torch.save(self.G.state_dict(), G_path)
		torch.save(self.D.state_dict(), D_path)

	def load(self, G_path, D_path):
		self.G.load_state_dict(torch.load(G_path))
		self.D.load_state_dict(torch.load(D_path))
