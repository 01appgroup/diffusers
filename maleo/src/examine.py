from datasets import load_dataset, load_from_disk, concatenate_datasets
import datasets
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_zeros(number, length):
	number_string = str(number)
	while len(number_string) < length:
		number_string = "0" + number_string
	return number_string

def test(example):
    image = example

rootpath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/"
for i in range(2, 4):
    suffix = pad_zeros(i, 3)
    filename = rootpath + "data-" + suffix + ".hf"
    dataset0 = load_from_disk(filename)
    ds = datasets.DatasetDict({"train": dataset0})
    print(ds['train'][0])
    print(i)
    try:
        ds = ds.map(test, num_proc=128)
    except:
        print("!!!!! Something wrong", i)