# to solve performance bottle neck of augmentation in cpu
from predef.nntracker import *
from utilitypack.utility import *

print("loading dataset")
datasetroot = r"dataset/"
dataset = labeldataset().init(
    os.path.join(datasetroot, r"largeEnoughToRecon/largeEnoughToRecon.zip"),
    os.path.join(datasetroot, r"largeEnoughToRecon/all.xlsx"),
    None,
    stdShape,
    augSteps=[
        labeldataset.AugSteps.affine,
        # labeldataset.AugSteps.rndln,
        # labeldataset.AugSteps.autoaug,
    ],
)

destSampleNum = 128
dest = r"dataset\testset"
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
    # name = DataCollector.geneName()
    filename = f"{i:05d}.png"
    names.append(filename)
    cv.imwrite(os.path.join(destSpl, filename), cv.cvtColor(spl, cv.COLOR_RGB2BGR) * 255)
    cv.imwrite(os.path.join(destLbl, filename), cv.cvtColor(lbl, cv.COLOR_RGB2BGR) * 255)
    prog.update(i)
save_list_to_xls(names, os.path.join(dest, "all.xlsx"))
