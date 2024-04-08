# %%
from predef.nntrackerdev_predef import *

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
        labeldataset.AugSteps.affine,
        labeldataset.AugSteps.randLine,
        labeldataset.AugSteps.autoaug,
        labeldataset.AugSteps.gausNoise,
    ],
)
print("load finished")


# %%
def NewAxis():
    fig, ax = plt.subplots()
    return ax


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
            NewAxis().imshow(item.lbl)
        Surf.append(surf)
        Wingspan.append(wingspan)
        prog.update(i)
    prog.setFinish()
    Surf = np.array(Surf)
    Wingspan = np.array(Wingspan)
    ws_surf_coef = (Wingspan**2) / Surf
    NewAxis().scatter(Surf, ws_surf_coef)


surfaceWingspanCoefFind(train_data)
# %%
