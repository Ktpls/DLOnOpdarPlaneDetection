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
    datasetroot = r"/kaggle/input/planedetectiondataset"

# %%
# nn def
modelpath = r"nntracker.pth"
model: nntracker_respi_spatialpositioning_head = getmodel(
    nntracker_respi_spatialpositioning_head(
        dropoutp=0,
    ),
    modelpath,
    device,
)
_ = setModuleFree(
    model.backbone,
    (
        # "layer3.0",
        "layer3.1",
        "layer4.0",
        "layer4.1",
    ),
)

# %%
# dataset
print("loading dataset")
datasets = {
    "LE2RE": NnTrackerDataset(r"LE2RE", r"LE2RE/all.xlsx"),
    "smallAug": NnTrackerDataset(r"smallAug", r"smallAug/all.xlsx"),
    "affined": NnTrackerDataset(r"affined", r"affined/all.xlsx"),
}
train_data = datasets["affined"]
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
print("load finished")

# %%
# dataloader
batch_size = 2
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
    outputperbatchnum=1000,
)

# %%
# save
model.save(modelpath)

# %%
# demo
model.demo(train_data)

# %%
