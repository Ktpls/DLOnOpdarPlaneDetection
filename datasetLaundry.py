# %%
from predef.nntrackerdev_predef import *
from utilitypack.util_torch import *
from utilitypack.util_pyplot import *
import torchvision.transforms.functional as TTF
# %%
# copied from nntrackerdev.py
datasetroot = r"dataset/"
print("loading dataset")
datasets = {
    "LE2REnh": NnTrackerDataset(r"LE2REnh/LE2REnh.zip", r"LE2REnh/all.xlsx", "zip"),
    "SmallAug": NnTrackerDataset(r"SmallAug/SmallAug.zip", r"SmallAug/all.xlsx", "zip"),
    "largeEnoughToRecon": NnTrackerDataset(
        r"largeEnoughToRecon/largeEnoughToRecon.zip",
        r"largeEnoughToRecon/all.xlsx",
        "zip",
    ),
    "origins_nntracker": NnTrackerDataset(
        r"origins_nntracker/origins_nntracker.zip",
        r"origins_nntracker/hardones.xlsx",
        "zip",
    ),
}
train_data = datasets["largeEnoughToRecon"]
train_data = labeldataset().init(
    os.path.join(datasetroot, train_data.path),
    os.path.join(datasetroot, train_data.sel),
    8192,
    train_data.datasettype,
    None,
    stdShape,
    augSteps=[
        #     labeldataset.AugSteps.affine,
        #     labeldataset.AugSteps.randLine,
        # labeldataset.AugSteps.autoaug,
        labeldataset.AugSteps.gausNoise,
    ],
)
print("load finished")


# %%


def surfaceWingspanCoefFind(dataset: labeldataset):
    numDesired = dataset.rawlength()
    prog = Progress(numDesired)
    Surf = list()
    Wingspan = list()
    surfall = 128 * 128
    for i in range(numDesired):
        item = dataset.items[i]
        surf = np.sum(item.lbl) / surfall + 1e-4
        wingspan = lbl2PlaneInfo(item.lbl)[3] + 1e-2
        coef = wingspan**2 / surf
        if coef > 20:
            print(item.name, coef, wingspan, surf)
            NewPyPlotAxis().imshow(item.lbl)
        Surf.append(surf)
        Wingspan.append(wingspan)
        prog.update(i)
    prog.setFinish()
    Surf = np.array(Surf)
    Wingspan = np.array(Wingspan)
    ws_surf_coef = (Wingspan**2) / Surf
    NewPyPlotAxis().scatter(Surf, ws_surf_coef)


surfaceWingspanCoefFind(train_data)

# %%


def checkAutoAugGood(dataset: labeldataset):
    """
    equalizer 1.0 None
    contrast 1.0 8
    """

    def myForward(
        self: torchvision.transforms.autoaugment.AutoAugment, img: torch.Tensor
    ) -> torch.Tensor:
        img = (img * 255).to(dtype=torch.uint8)
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = TTF.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.policies))

        op_meta = self._augmentation_space(10, (height, width))
        policies = self.policies[transform_id]
        for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[magnitude_id].item())
                    if magnitude_id is not None
                    else 0.0
                )
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                img = torchvision.transforms.autoaugment._apply_op(
                    img, op_name, magnitude, interpolation=self.interpolation, fill=fill
                )

        img = img.to(dtype=torch.float32) / 255
        return img, policies

    mppshape = np.array([6, 6])
    mpp = MassivePicturePlot(mppshape)
    numDesired = np.prod(mppshape) // 2
    prog = Progress(numDesired)

    for i in range(numDesired):
        item = dataset.items[dataset.rndIndex()]
        spl = item.spl
        spl = cv.cvtColor(spl, cv.COLOR_BGR2RGB)
        spl0 = spl * 0.5 + np.expand_dims(tensorimg2ndarray(item.lbl), -1) * 0.5
        spl = dataset.totensor(spl)
        spl, policies = myForward(dataset.augger, spl)
        p0, p1 = policies
        spl = tensorimg2ndarray(spl)
        mpp.toNextPlot()
        plt.title(p0)
        plt.imshow(spl0)
        prog.update(i)
        mpp.toNextPlot()
        plt.title(p1)
        plt.imshow(spl)
    prog.setFinish()


checkAutoAugGood(train_data)

# %%
