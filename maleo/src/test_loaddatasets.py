from datasets import load_dataset
import os

baseposition = '/nfs/users/zhangsan/datasets/laion-high-resolution-output/'
trainlist = []

df1=  os.path.join("/root/zhangsan/sd-pretrain/datasets/fengtangCCG", "**")
df2=  os.path.join("/root/zhangsan/sd-pretrain/datasets/pradaModa", "**")

dataset = load_dataset('imagefolder', data_files={'train': [df1,  df2]})

ds = dataset['train']
for d in ds:
    print(d)