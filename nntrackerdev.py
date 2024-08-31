# %%
# installing
# import os
# import sys


# def installLib(installpath, gitpath):
#     projPath = os.path.join(installpath, os.path.splitext(os.path.basename(gitpath))[0])
#     if not os.path.exists(projPath):
#         os.system(rf"git clone {gitpath} {projPath}")
#     else:
#         cwd = os.getcwd()
#         os.chdir(projPath)
#         os.system(rf"git pull")
#         os.chdir(cwd)
#     if projPath not in sys.path:
#         sys.path.append(projPath)

# installLib(r"/kaggle/working", "https://github.com/Ktpls/pyinclude.git")
# installLib(r"/kaggle/working", "https://github.com/Ktpls/DLOnOpdarPlaneDetection.git")
# os.chdir(r"/kaggle/working")
# #!rm "/kaggle/working/DLOnOpdarPlaneDetection/nntracker.pth"

# %%
# basics
from predef.nntrackerdev_predef import *

import_or_reload("predef.nntracker")
from predef.nntracker import *

RunOnWtUtilityEnviroment = True
if RunOnWtUtilityEnviroment:
    datasetroot = r"dataset/"
else:
    datasetroot = r"/kaggle/input/nntrackerle2re"

# %%
# nn def
modelpath = r"nntracker.pth"
model: nntracker_respi = getmodel(
    nntracker_respi_mnv3s(
        dropoutp=0,
    ),
    modelpath,
    device,
)
_ = setModuleFree(
    model.backbone,
    (
        "features.9",
        "features.10",
        "features.11",
        "features.12",
    ),
)

# %%
# dataset
print("loading dataset")
datasets = {
    "LE2RE": NnTrackerDataset(r"LE2RE", r"LE2RE/all.xlsx"),
    "smallAug": NnTrackerDataset(r"smallAug", r"smallAug/all.xlsx"),
    "affined": NnTrackerDataset(r"affined", r"affined/all.xlsx"),
    "testset": NnTrackerDataset(r"testset", r"testset/all.xlsx"),
    "largeEnoughToRecon": NnTrackerDataset(
        r"largeEnoughToRecon/largeEnoughToRecon.zip", r"largeEnoughToRecon/all.xlsx"
    ),
}
train_data = datasets["LE2RE"]
train_data = labeldataset().init(
    path=os.path.join(datasetroot, train_data.path),
    selection=os.path.join(datasetroot, train_data.sel),
    stdShape=stdShape,
    device=device,
    augSteps=[
        # labeldataset.AugSteps.affine,
        # labeldataset.AugSteps.rndln,
        # labeldataset.AugSteps.autoaug,
        # labeldataset.AugSteps.gausNoise,
    ],
)
test_data = datasets["largeEnoughToRecon"]
test_data = labeldataset().init(
    path=os.path.join(datasetroot, test_data.path),
    selection=os.path.join(datasetroot, test_data.sel),
    stdShape=stdShape,
    device=device,
    augSteps=[],
)
print("load finished")

# %%
# dataloader
batch_size = 4
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# %%
# train
trainpipe.train(
    train_dataloader,
    torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    ),
    lambda *p: model.trainprogress(*p),
    epochnum=10,
    outputperbatchnum=500,
)

# %%
# save
model.save(modelpath)

# %%
# demo
model.demo(train_data)