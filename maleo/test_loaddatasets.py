from datasets import load_dataset, load_from_disk, concatenate_datasets
import datasets
import os
import sys
# baseposition = '/nfs/users/zhangsan/datasets/laion-high-resolution-output/'
# trainlist = []

# df1=  os.path.join("/root/zhangsan/sd-pretrain/datasets/fengtangCCG", "**")
# df2=  os.path.join("/root/zhangsan/sd-pretrain/datasets/pradaModa", "**")

# dataset = load_dataset('imagefolder', data_files={'train': [df1,  df2]})
# print(dataset)
# ds = dataset['train']
# print(ds)


def test(example):
    image = example['image']
    #print(image)

dataset0 = load_from_disk("/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/data-005.hf")
ds = datasets.DatasetDict({"train": dataset0})
#ds = ds.map(test, num_proc=128)  # num_proc for multiprocessing


# for i, dataset in enumerate(dataset0):
#     print(i)
# sys.exit()
exclude_idx = [40275]
print("AAAAAAA")
dataset1 = dataset0.select(
    (
        i for i in range(len(dataset0)) 
        if i not in set(exclude_idx)
    )
)
print("BBBBBBB")
dataset1.save_to_disk("/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/data-005A.hf")

# ds = datasets.DatasetDict({"train": dataset1})
# ds = ds.map(test, num_proc=128)  # num_proc for multiprocessing

# dataset2 = load_from_disk("/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/data-005.hf")
# dataset3 = load_from_disk("/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/data-006.hf")
# dataset4 = load_from_disk("/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/data-007.hf")


# dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4])
# d = datasets.DatasetDict({"train": dataset})
# print(d)