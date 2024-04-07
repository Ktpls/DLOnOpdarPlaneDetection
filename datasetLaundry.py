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
        r"largeEnoughToRecon/largeEnoughToRecon",
        r"largeEnoughToRecon/all.xlsx",
        "fld",
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


def surfaceWingspanCoefFind(dataset: labeldataset):
    numDesired = dataset.rawlength()
    prog = Progress(numDesired)
    records = list()
    surfall = 128 * 128
    for i in range(numDesired):
        item = dataset.items[i]
        surf = np.sum(item.lbl) / surfall
        wingspan = lbl2PlaneInfo(item.lbl)[3]
        # over = wingspan - ((0.5 - 0.3) / 0.03 * surf + 0.3)
        # if over > 0:
        #     print(item.name, over, wingspan, surf)
        records.append([(surf), (wingspan)])
        prog.update(i)
    prog.setFinish()
    records = np.array(records)
    surf = records[:, 0]
    wingspan = records[:, 1]
    ws_surf_coef = (surf) / (wingspan**2 + 1)
    plt.scatter(wingspan, ws_surf_coef)


surfaceWingspanCoefFind(train_data)
# %%
