# %% utilkaggle
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TTF
import cv2 as cv
import dataclasses
import functools
import itertools
import matplotlib.pyplot as plt
import numpy as np
import openpyxl as opx
import os
import platform
import time
import torch
import torchvision
import typing
import enum


def NormalizeImgToChanneled_CvFormat(m:cv.Mat):
    return m if len(m.shape) == 3 else m.reshape(m.shape + (1,))

def Xls2ListList(path=None, sheetname=None, killNones=True):
    if path is None:
        path = r"eles.in.xlsx"
    xls = opx.load_workbook(path)
    if sheetname is None:
        sheet = xls.active
    else:
        sheet = xls[sheetname]

    ret = [[ele.value for ele in ln] for ln in (sheet.rows)]
    if killNones:
        ret = [l for l in ret if any([e is not None for e in l])]
    return ret


def AllFileIn(path, includeFileInSubDir=True):
    import os

    ret = []
    for dirpath, dir, file in os.walk(path):
        if not includeFileInSubDir and dirpath != path:
            continue
        ret.extend([os.path.join(dirpath, f) for f in file])
    return ret


def batchsizeof(tensor):
    return tensor.shape[0]


def getDeviceInfo():
    # Get CPU info
    cpu_info = platform.processor()
    num_cores = os.cpu_count()
    cpu_frequency = "Not available in Python standard library"

    # Get GPU info
    try:
        gpu_info = torch.cuda.get_device_name()
    except:
        gpu_info = "No GPU detected"

    return (
        f"CPU Info: {cpu_info}, Cores: {num_cores}, Frequency: {cpu_frequency}\n"
        + f"GPU Info: {gpu_info}"
    )

class nestedPyPlot:

    def __init__(self, outtershape, innershape, fig) -> None:
        self.oshape = np.array(outtershape)
        self.ishape = np.array(innershape)
        self.realsize = self.oshape * self.ishape
        self.fig = fig

    def subplot(self, o, i):
        maincoor = np.array((int(o / self.oshape[1]), o % self.oshape[1]))
        subcoor = np.array((int(i / self.ishape[1]), i % self.ishape[1]))
        realcoor = self.ishape * maincoor + subcoor
        # plt.subplot(self.realsize[0], self.realsize[1], realcoor[0]
        #             * self.realsize[1]+realcoor[1]+1)
        ax = self.fig.add_subplot(
            self.realsize[0],
            self.realsize[1],
            realcoor[0] * self.realsize[1] + realcoor[1] + 1,
        )
        return ax


def setModel(model, path=None, device="cpu"):
    import os

    if path is None:
        print(f"Path==None")
        return model
    elif not os.path.exists(path):
        print(f"Warning: Path {path} not exist. Set model default")
        return model
    else:
        print(f"Loading existed nn {path}")
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        # torch.load with map_location=torch.device('cpu')
        return model


def tensorimg2ndarray(m):
    m = np.array(m)
    m = np.moveaxis(m, -3, -1)
    return m


def spectrumDecompose(s, psize):
    if type(s) is int:
        s = torch.tensor([s])
    batchsize = s.shape[0]
    s = s.reshape((batchsize, 1))
    return torch.zeros([batchsize, psize], dtype=torch.float).scatter_(
        dim=-1, index=s, src=torch.ones_like(s, dtype=torch.float)
    )


def batchsizeof(tensor):
    return tensor.shape[0]


class nestedPyPlot:
    def __init__(self, outtershape, innershape, fig) -> None:
        self.oshape = np.array(outtershape)
        self.ishape = np.array(innershape)
        self.realsize = self.oshape * self.ishape
        self.fig = fig

    def subplot(self, o, i):
        maincoor = np.array((int(o / self.oshape[1]), o % self.oshape[1]))
        subcoor = np.array((int(i / self.ishape[1]), i % self.ishape[1]))
        realcoor = self.ishape * maincoor + subcoor
        # plt.subplot(self.realsize[0], self.realsize[1], realcoor[0]
        #             * self.realsize[1]+realcoor[1]+1)
        ax = self.fig.add_subplot(
            self.realsize[0],
            self.realsize[1],
            realcoor[0] * self.realsize[1] + realcoor[1] + 1,
        )
        return ax


def setModule(model, path=None, device=None):
    import os

    if device is None:
        device = "cpu"

    if path is None:
        print(f"Path==None")
    elif not os.path.exists(path):
        print(f"Warning: Path {path} not exist. Set model default")
    else:
        print(f"Loading existed nn {path}")
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model.to(device)


def tensorimg2ndarray(m):
    m = m.cpu().numpy()
    m = np.moveaxis(m, -3, -1)
    return m


class skiper(torch.nn.Module):
    def __init__(self, component, n_i, n_o) -> None:
        super().__init__()
        self.component = component
        self.combiner = torch.nn.Conv2d(n_i + n_o, n_o, 1)

    def forward(self, m):
        # [b,c,h,w]
        processed = self.component.forward(m)
        c = torch.concat([processed, m], dim=-3)
        result = self.combiner.forward(c)
        return result


class cbr(torch.nn.Module):
    def __init__(self, n_i, n_o, n_c) -> None:
        super().__init__()
        self.component = torch.nn.Sequential(
            torch.nn.Conv2d(n_i, n_o, n_c, padding="same", bias=False),
            torch.nn.BatchNorm2d(n_o),
            torch.nn.LeakyReLU(),
        )

    def forward(self, m):
        # [b,c,h,w]
        return self.component.forward(m)


class cbrps(torch.nn.Module):
    # input chan, output chan, convolve size, pooling size
    # n_o should be like 2*n, cuz maxpool will be concated with former output
    def __init__(self, n_i, n_o, n_c, n_p) -> None:
        super().__init__()
        self.component = torch.nn.Sequential(
            torch.nn.Conv2d(n_i, n_o, n_c, padding="same", bias=False),
            torch.nn.BatchNorm2d(n_o),
            torch.nn.LeakyReLU(),
            skiper(torch.nn.MaxPool2d(n_p, stride=1, padding=int(n_p / 2)), n_o, n_o),
        )

    def forward(self, m):
        # [b,c,h,w]
        return self.component.forward(m)


class inception(torch.nn.Module):
    def __init__(
        self,
        infeat,
        outfeat11,
        outfeatpool,
        outfeat33,
        outfeat55,
        isbn=True,
        version=None,
    ) -> None:
        super().__init__()
        self.infeat = infeat
        self.outfeat11 = outfeat11
        self.outfeatpool = outfeatpool
        self.outfeat33 = outfeat33
        self.outfeat55 = outfeat55
        self.isbn = isbn
        if version is None:
            version = "v2"
        self.version = version
        if version == "v2":
            self.path11 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat11, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.pathpool = torch.nn.Sequential(
                torch.nn.MaxPool2d(3, stride=1, padding=1),
                torch.nn.Conv2d(infeat, outfeatpool, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path33 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, infeat, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(infeat, outfeat33, 3, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path55 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, infeat, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(infeat, outfeat55, 3, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, 3, padding="same"),
                torch.nn.LeakyReLU(),
            )
        elif version == "v3":
            self.path11 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat11, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.pathpool = torch.nn.Sequential(
                torch.nn.MaxPool2d(3, stride=1, padding=1),
                torch.nn.Conv2d(infeat, outfeatpool, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path33 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat55, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat33, [1, 3], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat33, outfeat33, [3, 1], padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path55 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat55, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [1, 3], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [3, 1], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [1, 3], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [3, 1], padding="same"),
                torch.nn.LeakyReLU(),
            )
        else:
            raise ValueError(f"{version} not supported")
        if isbn is not None and isbn:
            self.bn = torch.nn.BatchNorm2d(
                outfeat11 + outfeatpool + outfeat33 + outfeat55
            )
        else:
            self.bn = None

    @staticmethod
    def even(infeat, outfeat, bn=None, version=None):
        assert outfeat % 4 == 0
        outfeatby4 = outfeat // 4
        return inception(
            infeat, outfeatby4, outfeatby4, outfeatby4, outfeatby4, bn, version
        )

    def forward(self, m):
        o = torch.concat(
            [self.path11(m), self.pathpool(m), self.path33(m), self.path55(m)], dim=-3
        )
        if self.bn is not None:
            o = self.bn(o)
        return o  # channel


class res_through(torch.nn.Module):
    _modules = {}

    def __init__(self, *components, combiner=None) -> None:
        super().__init__()
        self.components = components
        for idx, module in enumerate(components):
            self.add_module(str(idx), module)
        if combiner is None:

            def combiner_add(last, current):
                return last + current

            combiner = combiner_add
        self.combiner = combiner

    def forward(self, m):
        o = m
        for i, l in enumerate(self.components):
            ret = l(o)
            o = self.combiner(o, ret)
        return o


class ModuleFunc(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class trainpipe:
    @staticmethod
    def train(
        dataloader,
        optimizer,
        calcLossFoo,
        epochnum=6,
        outputperbatchnum=100,
        customSubOnOutput=None,
    ):
        epochs = epochnum
        start_time = time.time()
        for ep in range(epochs):
            print(f"Epoch {ep+1}")
            print("-------------------------------")

            # train
            for batch, datatuple in enumerate(dataloader):
                loss = calcLossFoo(batch, datatuple)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch % outputperbatchnum == 0:
                    end_time = time.time()
                    print(f"Batch {batch}/{len(dataloader)}")
                    print(
                        f"Training speed: {outputperbatchnum/(end_time-start_time):>5f} batches per second"
                    )
                    aveloss = loss.item() / batchsizeof(datatuple[0])
                    print(f"Average loss: {aveloss:>7f}")
                    if customSubOnOutput is not None:
                        customSubOnOutput(batch, aveloss)
                    start_time = time.time()

        # win32api.Beep(1000, 1000)
        print("Done!")

EPS = 1e-10
import time
class perf_statistic:
    """
    calculate the time past between start() to now, directly by perf_counter()-starttime
    record all accumulated time before start(), but uncleared after stop()
    so start and stop are also playing roles as resume and pause
    countcycle() will increase the cycle count, helping to calculate average time in a loop-like task
    clear() will clear all accumulated time, stops counting
    """

    def __init__(self, startnow=False):
        self.clear()
        if startnow:
            self.start()

    def clear(self):
        self._starttime = None
        self._stagedtime = 0
        self._cycle = 0
        return self

    def start(self):
        self._starttime = time.perf_counter()
        return self

    def countcycle(self):
        self._cycle += 1
        return self

    def stop(self):
        if self._starttime is None:
            return
        self._stagedtime += self._timeCurrentlyCounting()
        self._starttime = None
        return self

    def time(self):
        return self._stagedtime + self._timeCurrentlyCounting()

    def aveTime(self):
        return self.time() / (self._cycle if self._cycle > 0 else 1)

    def _timeCurrentlyCounting(self):
        return (
            time.perf_counter() - self._starttime if self._starttime is not None else 0
        )
class Progress:
    """
    irreversable!
    """

    def __init__(self, total: float, cur=0, printPercentageStep: float = 0.1) -> None:
        self.total = total
        self.nowStage = 0
        self.printPercentageStep = printPercentageStep
        self.cur = cur
        self.ps = perf_statistic()

    def update(self, current: float) -> None:
        self.cur = current
        while True:
            if current / self.total > self.nowStage * self.printPercentageStep:
                self.nowStage += 1
                self.ps.stop()
                if current > 1:
                    # not the first time
                    instantSpeed = (self.printPercentageStep * self.total) / (
                        self.ps.time() + EPS
                    )
                else:
                    instantSpeed = 1
                self.ps.clear().start()
                print(
                    f"{100 * current / self.total:>3.2f}% of {self.total}, {instantSpeed:.2f}it/s",
                    end="\n",
                )
            else:
                break

    def setFinish(self):
        self.update(self.total)
        

def Deduplicate(l: list):
    return list(set(l))
class Stream:
    content: list
    actions: list

    def __init__(self, iter: list | tuple | dict) -> None:
        if isinstance(iter, (list, tuple)):
            self.content = iter
        elif isinstance(iter, dict):
            self.content = iter.items()
        else:
            raise TypeError("iter must be list|tuple|dict")

    def sort(self, pred: typing.Callable[[typing.Any, typing.Any], int]):
        self.content.sort(key=functools.cmp_to_key(pred))
        return self

    def peek(self, pred: typing.Callable[[typing.Any], None]):
        for i in self.content:
            pred(i)
        return self

    def filter(self, pred: typing.Callable[[typing.Any], bool]):
        self.content = list(filter(pred, self.content))
        return self

    def map(self, pred: typing.Callable[[typing.Any], typing.Any]):
        self.content = list(map(pred, self.content))
        return self

    def flatMap(self, pred: "typing.Callable[[typing.Any],Stream]"):
        self.content = list(
            itertools.chain.from_iterable([s.content for s in map(pred, self.content)])
        )
        return self

    def distinct(self):
        self.content = Deduplicate(self.content)
        return self

    class Collector:
        def __init__(self, collectImpl):
            self.collectImpl = collectImpl

        def do(self, stream):
            return self.collectImpl(stream)

        @staticmethod
        def toList():
            return Stream.Collector(lambda stream: list(stream.content))

        @staticmethod
        def toDict(keyPred, valuePred):
            return Stream.Collector(
                lambda stream: {keyPred(i): valuePred(i) for i in stream.content}
            )

        @staticmethod
        def groupBy(keyPred):
            return Stream.Collector(
                lambda stream: {
                    key: list(group)
                    for key, group in itertools.groupby(
                        sorted(stream.content, key=keyPred), key=keyPred
                    )
                }
            )

    def collect(self, collector: "Stream.Collector"):
        return collector.do(self)