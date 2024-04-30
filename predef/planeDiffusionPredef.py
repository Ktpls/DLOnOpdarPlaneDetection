from utilitypack.util_solid import *
from utilitypack.util_import import *
from utilitypack.util_np import *
from utilitypack.util_ocv import *
from utilitypack.util_torch import *


class T2I(torch.nn.Module):
    resolution = [28, 28]
    chanInput = 1
    nPrompt = 10
    nSigma = 2
    chanPrompt = 4
    chanImgEncoded = 4
    subsampleRate0_1 = 2
    subsampleRate1_2 = 2
    chanFeature0 = 32
    chanFeature1 = 64
    chanFeature2 = 128
    featExport0 = 1
    featExport1 = 4
    featExport2 = 7

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        resolution = self.resolution
        chanInput = self.chanInput
        nPrompt = self.nPrompt
        chanPrompt = self.chanPrompt
        chanImgEncoded = self.chanImgEncoded
        nSigma = self.nSigma
        subsampleRate0_1 = self.subsampleRate0_1
        subsampleRate1_2 = self.subsampleRate1_2
        chanFeature0 = self.chanFeature0
        chanFeature1 = self.chanFeature1
        chanFeature2 = self.chanFeature2
        self.promptEncoder = torch.nn.Sequential(
            torch.nn.Linear(nPrompt + nSigma, 128),
            torch.nn.Hardswish(),
            torch.nn.Linear(128, chanPrompt * np.prod(resolution)),
            torch.nn.Hardswish(),
            ModuleFunc(lambda x: x.reshape([-1, chanPrompt, *resolution])),
        )
        self.imgEncoder = torch.nn.Sequential(
            ConvBnHs(chanInput, chanImgEncoded, 7),
            ConvBnHs(chanImgEncoded, chanImgEncoded),
        )
        self.featureExtractor = torch.nn.Sequential(
            ConvBnHs(chanImgEncoded + chanPrompt, chanFeature0),
            res_through(
                ConvBnHs(chanFeature0, chanFeature0),
                ConvBnHs(chanFeature0, chanFeature0),
            ),
            torch.nn.MaxPool2d(subsampleRate0_1, subsampleRate0_1),
            ConvBnHs(chanFeature0, chanFeature1),
            res_through(
                ConvBnHs(chanFeature1, chanFeature1),
                ConvBnHs(chanFeature1, chanFeature1),
            ),
            torch.nn.MaxPool2d(subsampleRate1_2, subsampleRate1_2),
            ConvBnHs(chanFeature1, chanFeature2),
            res_through(
                ConvBnHs(chanFeature2, chanFeature2),
                ConvBnHs(chanFeature2, chanFeature2),
            ),
        )
        self.Path0 = torch.nn.Sequential(
            res_through(
                ConvBnHs(chanFeature0, chanFeature0),
                ConvBnHs(chanFeature0, chanFeature0),
            ),
        )
        self.Path1 = torch.nn.Sequential(
            res_through(
                ConvBnHs(chanFeature1, chanFeature1),
                ConvBnHs(chanFeature1, chanFeature1),
            ),
        )
        self.Path2 = torch.nn.Sequential(
            res_through(
                ConvBnHs(chanFeature2, chanFeature2),
                ConvBnHs(chanFeature2, chanFeature2),
            ),
        )
        self.path2topath0 = torch.nn.Upsample(
            scale_factor=subsampleRate1_2 * subsampleRate0_1, mode="bilinear"
        )
        self.path1topath0 = torch.nn.Upsample(
            scale_factor=subsampleRate0_1, mode="bilinear"
        )
        self.output = torch.nn.Sequential(
            ConvBnHs(chanFeature0 + chanFeature1 + chanFeature2, chanFeature0),
            res_through(
                ConvBnHs(chanFeature0, chanFeature0),
                ConvBnHs(chanFeature0, chanFeature0),
            ),
            ConvBnHs(chanFeature0, chanInput),
        )

    def forward(self, noiimg, prompt, sigmaLower, sigmaUpper):
        prompt = torch.concat(
            [prompt, sigmaLower.unsqueeze(-1), sigmaUpper.unsqueeze(-1)], dim=-1
        )
        imgEncoded = self.imgEncoder(noiimg)
        promptEncoded = self.promptEncoder(prompt)
        x = torch.cat([imgEncoded, promptEncoded], dim=-3)
        for i, m in enumerate(self.featureExtractor):
            x = m(x)
            if i == self.featExport0:
                x0 = x
            if i == self.featExport1:
                x1 = x
            if i == self.featExport2:
                x2 = x
        x0 = self.Path0(x0)
        x1 = self.Path1(x1)
        x2 = self.Path2(x2)
        x = torch.concat([x0, self.path1topath0(x1), self.path2topath0(x2)], dim=-3)
        x = self.output(x)
        return x


class DiffusionUtil:
    @staticmethod
    def noiseLike(x, sigma=1, mu=0):
        return torch.randn_like(x) * sigma + mu

    @staticmethod
    def noisedImg(img, sigma=1, snr=0.5):
        noise = DiffusionUtil.noiseLike(img, sigma)
        noiimg = noise * (1 - snr) + img * snr
        return noiimg


def viewDatasetVal(sampleNum, FetchX):
    with torch.no_grad():
        samples = []
        prog = Progress(sampleNum)
        for i in range(sampleNum):
            img = FetchX(i)
            samples.append(img)
            prog.update(i)
        prog.setFinish()
        samples = torch.stack(samples)
        std, mean = torch.std_mean(samples)
        print(f"{mean=:.3f}, {std=:.3f}")
