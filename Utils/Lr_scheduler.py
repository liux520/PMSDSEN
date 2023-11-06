##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
import bisect
import time

import numpy as np

from Utils.print_utils import print_info_message

__all__ = ['LR_Scheduler', 'LR_Scheduler_Head', 'CyclicLR', 'HybirdLR']


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, quiet=False):
        self.mode = mode
        self.quiet = quiet
        if not quiet:
            print_info_message('Using {} LR scheduler with warm-up epochs of {}!'.format(self.mode, warmup_epochs))
        if mode == 'step':
            assert lr_step
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == 'linear':
            lr = self.base_lr - (self.base_lr * (1.0 * T / self.total_iters))
        else:
            raise NotImplemented
        if epoch > self.epoch and (epoch == 0 or best_pred > 0.0):
            if not self.quiet:
                print_info_message('\n==> {} Epoches {}, learning rate = {:.6f}, previous best = {:.4f}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr


class LR_Scheduler_Head(LR_Scheduler):
    """Incease the additional head LR to be 10 times"""

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"


# ============================================

class CyclicLR(object):
    '''
    CLass that defines cyclic learning rate with warm restarts that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    See https://arxiv.org/abs/1811.11431 for more details
    '''

    def __init__(self, min_lr=0.1, cycle_len=5, steps=[51, 101, 131, 161, 191, 221, 251, 281], gamma=0.5, step=True):
        super(CyclicLR, self).__init__()
        assert len(steps) > 0, 'Please specify step intervals.'
        assert 0 < gamma <= 1, 'Learing rate decay factor should be between 0 and 1'
        self.min_lr = min_lr  # minimum learning rate
        self.m = cycle_len
        self.steps = steps
        self.warm_up_interval = 1  # we do not start from max value for the first epoch, because some time it diverges
        self.counter = 0
        self.decayFactor = gamma  # factor by which we should decay learning rate
        self.count_cycles = 0
        self.step_counter = 0
        self.stepping = step

    def __call__(self, optimizer, i, epoch, best_miou):
        if epoch % self.steps[self.step_counter] == 0 and epoch > 1 and self.stepping:
            self.min_lr = self.min_lr * self.decayFactor
            self.count_cycles = 0
            if self.step_counter < len(self.steps) - 1:
                self.step_counter += 1
            else:
                self.stepping = False
        current_lr = self.min_lr
        # warm-up or cool-down phase
        if self.count_cycles < self.warm_up_interval:
            self.count_cycles += 1
            # We do not need warm up after first step.
            # so, we set warm up interval to 0 after first step
            if self.count_cycles == self.warm_up_interval:
                self.warm_up_interval = 0
        else:
            # Cyclic learning rate with warm restarts
            # max_lr (= min_lr * step_size) is decreased to min_lr using linear decay before
            # it is set to max value at the end of cycle.
            if self.counter >= self.m:
                self.counter = 0
            current_lr = round((self.min_lr * self.m) - (self.counter * self.min_lr), 5)
            self.counter += 1
            self.count_cycles += 1
        print_info_message('\n==> {} Epoches {}, learning rate = {:.6f}, previous best = {:.4f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, current_lr, best_miou))

        self._adjust_learning_rate(optimizer, current_lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

        # return current_lr

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Min. base LR: {}\n'.format(self.min_lr)
        fmt_str += '    Max. base LR: {}\n'.format(self.min_lr * self.m)
        fmt_str += '    Step interval: {}\n'.format(self.steps)
        fmt_str += '    Decay lr at each step by {}\n'.format(self.decayFactor)
        return fmt_str


class HybirdLR(object):
    def __init__(self, base_lr, clr_max, max_epochs, cycle_len=5):
        super(HybirdLR, self).__init__()
        self.linear_epochs = max_epochs - clr_max + 1
        steps = [clr_max]
        self.clr = CyclicLR(min_lr=base_lr, cycle_len=cycle_len, steps=steps, gamma=1)
        # self.decay_lr = LinearLR(base_lr=base_lr, max_epochs=self.linear_epochs)
        self.decay_lr = PolyLR(base_lr=base_lr, max_epochs=self.linear_epochs, power=0.9)
        self.cyclic_epochs = clr_max

        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.clr_max = clr_max
        self.cycle_len = cycle_len

    def __call__(self, optimizer, i, epoch, best_miou):
        if epoch < self.cyclic_epochs:
            curr_lr = self.clr(optimizer, i, epoch, best_miou)
        else:
            curr_lr = self.decay_lr.step(epoch - self.cyclic_epochs + 1)
        curr_lr = round(curr_lr, 6)
        print_info_message('\n==> {} Epoches {}, learning rate = {:.6f}, previous best = {:.4f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, curr_lr, best_miou))
        self._adjust_learning_rate(optimizer, curr_lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

        # return round(curr_lr, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Cycle with length of {}: {}\n'.format(self.cycle_len, int(self.clr_max / self.cycle_len))
        fmt_str += '    Base LR with {} cycle length: {}\n'.format(self.cycle_len, self.base_lr)
        fmt_str += '    Cycle with length of {}: {}\n'.format(self.linear_epochs, 1)
        fmt_str += '    Base LR with {} cycle length: {}\n'.format(self.linear_epochs, self.base_lr)
        return fmt_str


class FixedMultiStepLR(object):
    '''
        Fixed LR scheduler with steps
    '''

    def __init__(self, base_lr=0.1, steps=[30, 60, 90], gamma=0.1, step=True):
        super(FixedMultiStepLR, self).__init__()
        assert len(steps) > 1, 'Please specify step intervals.'
        self.base_lr = base_lr
        self.steps = steps
        self.decayFactor = gamma  # factor by which we should decay learning rate
        self.stepping = step
        print('Using Fixed LR Scheduler')

    def step(self, epoch):
        return round(self.base_lr * (self.decayFactor ** bisect.bisect(self.steps, epoch)), 5)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Base LR: {}\n'.format(self.base_lr)
        fmt_str += '    Step interval: {}\n'.format(self.steps)
        fmt_str += '    Decay lr at each step by {}\n'.format(self.decayFactor)
        return fmt_str


class LinearLR(object):
    def __init__(self, base_lr, max_epochs):
        super(LinearLR, self).__init__()
        self.base_lr = base_lr
        self.max_epochs = max_epochs

    def step(self, epoch):
        curr_lr = self.base_lr - (self.base_lr * (epoch / (self.max_epochs)))
        return round(curr_lr, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR: {}\n'.format(self.base_lr)
        return fmt_str


class CosineLR(object):
    def __init__(self, base_lr, max_epochs):
        super(CosineLR, self).__init__()
        self.base_lr = base_lr
        self.max_epochs = max_epochs

    def step(self, epoch):
        return round(self.base_lr * (1 + math.cos(math.pi * epoch / self.max_epochs)) / 2, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR : {}\n'.format(self.base_lr)
        return fmt_str


class PolyLR(object):
    '''
        Polynomial LR scheduler with steps
    '''

    def __init__(self, base_lr, max_epochs, power=0.99):
        super(PolyLR, self).__init__()
        # assert 0 < power < 1
        self.base_lr = base_lr
        self.power = power
        self.max_epochs = max_epochs

    def step(self, epoch):
        curr_lr = self.base_lr * (1 - (float(epoch) / self.max_epochs)) ** self.power
        return round(curr_lr, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR: {}\n'.format(self.base_lr)
        fmt_str += '    Power: {}\n'.format(self.power)
        return fmt_str


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    step = np.arange(1, 100, 90)
    print(step)  # [51, 161, 201]
    step_size = 61
    step_sizes = [step_size * i for i in range(1, int(math.ceil(200 / step_size)))]
    plt.plot(step_sizes)
    plt.title(f'[step_size * i for i in range(1, int(math.ceil(200 / step_size=61)))]')
    plt.show()
    # lr_scheduler = HybirdLR(0.005,61,300)#FixedMultiStepLR(0.5,)#CyclicLR(min_lr=0.005, steps=step, gamma=1)# #LinearLR(0.5,100)#PolyLR(0.5,100,power=2)
    # lr_list=[]
    # for i in range(300):
    #     lr = lr_scheduler.step(i)
    #     lr_list.append(lr)
    # plt.plot(range(300),lr_list)
    # print(lr_list)
    # plt.title(f'HybirdLR clr_max=61 min_lr=0.005 cycle_len=5')
    # plt.show()

    # max_epochs = 100
    # lrSched = PolyLR(0.007, max_epochs=max_epochs)
    # #lrSched = CosineLR(0.01, max_epochs=200)
    # #for i in range(max_epochs):
    # #    print(i, lrSched.step(i))
    # #exit()
    # lrSched = HybirdLR(0.007, clr_max=51, max_epochs=max_epochs)
    # for i in range(max_epochs):
    #     print(i, lrSched.step(i))
    # exit()
    # lrSched = PolyLR(0.007, max_epochs=50)
    # for i in range(50):
    #     print(i, lrSched.step(i))
    #
    # max_epochs = 240
    # lrSched = CyclicLR(min_lr=0.01, steps=[51, 161, 201], gamma=0.1)
    # print(lrSched)
    # for i in range(max_epochs):
    #     print(i, lrSched.step(i))
