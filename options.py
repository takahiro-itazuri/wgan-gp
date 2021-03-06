import os
import argparse
import json
import torch

class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# model
		parser.add_argument('--G_act_type', type=str, default='relu', help='activation type of generator')
		parser.add_argument('--G_norm_type', type=str, default='batchnorm', help='normalization type of generator')
		parser.add_argument('--C_act_type', type=str, default='relu', help='activation type of critic')
		parser.add_argument('--C_norm_type', type=str, default='layernorm', help='normalization type of critic')
		parser.add_argument('--condition', action='store_true', default=False, help='conditional version')
		# dataset
		parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 | cifar100 | stl10')
		parser.add_argument('--num_workers', type=int, default=12, help='number of workers')
		# hyperparameter
		parser.add_argument('--batch_size', type=int, default=64, help='batch size')
		# log
		parser.add_argument('--log_dir', type=str, required=True, help='log directory')
		# GPU
		parser.add_argument('--use_gpu', action='store_true', default=False, help='GPU mode ON/OFF')
		parser.add_argument('--device_id', type=int, default=None, help='GPU device ID')

		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.log_dir, exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')

	def parse(self):
		opt = self.gather_options()
		if opt.condition:
			opt.G_norm_type = 'conditional_batchnorm'

		self.print_options(opt)

		if self.__class__.__name__ == 'TrainOptions':
			opt.train = True
		else:
			opt.train = False

		if opt.dataset == 'cifar10':
			opt.num_classes = 10
			opt.input_size = 32
		elif opt.dataset == 'cifar100':
			opt.num_classes = 100
			opt.input_size = 32
		elif opt.dataset == 'stl10':
			opt.num_classes = 10
			opt.input_size = 64

		# GPU
		if opt.use_gpu and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			if opt.device_id:
				opt.device = torch.device('cuda:{}'.format(opt.device_id))
			else:
				opt.device = torch.device('cuda')
		else:
			opt.device = torch.device('cpu')

		self.opt = opt
		return self.opt

class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# hyperparameter
		parser.add_argument('--num_itrs', type=int, default=50000, help='number of iterations')
		parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for Adam')
		parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
		parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
		parser.add_argument('--num_critic', type=int, default=5, help='number of critic iteration for one generator iteration')
		parser.add_argument('--lambda_gp', type=float, default=10, help='coefficient of gradient penalty')
		# log
		parser.add_argument('--checkpoint', type=int, default=5000, help='checkpoint iteration')
		return parser

class TestOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# model
		parser.add_argument('--weight', type=str, required=True, help='path to model weight')
		return parser
