# to solve performance bottle neck of augmentation in cpu
from predef.nntracker import *
from utilitypack.utility import *

print("loading dataset")
datasetroot = r"dataset/"
dataset = labeldataset().init(
    os.path.join(datasetroot, r"largeEnoughToRecon/largeEnoughToRecon.zip"),
    os.path.join(datasetroot, r"largeEnoughToRecon/all.xlsx"),
    8192,
    "zip",
    None,
    stdShape,
    augSteps=[
        labeldataset.AugSteps.affine,
        labeldataset.AugSteps.randLine,
        labeldataset.AugSteps.autoaug,
    ],
)

destSampleNum = 8192
dest = r"dataset\LE2REnh"
destSpl = os.path.join(dest, "spl")
destLbl = os.path.join(dest, "lbl")
names = list()
EnsureDirectoryExists(destSpl)
EnsureDirectoryExists(destLbl)
prog = Progress(destSampleNum)
for i in range(destSampleNum):
    index = dataset.rndIndex()
    item = dataset.items[index]
    item = dataset.dataAug(item)
    spl, lbl = item.spl, item.lbl
    name = DataCollector.geneName()
    filename = f"{name}.png"
    names.append(filename)
    cv.imwrite(os.path.join(destSpl, filename), spl * 255)
    cv.imwrite(os.path.join(destLbl, filename), lbl * 255)
    prog.update(i)
save_list_to_xls(names, os.path.join(dest, "all.xlsx"))
