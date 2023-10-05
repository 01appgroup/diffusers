from datasets import load_dataset, load_from_disk
import time
import os
import json
import glob
from datasets import Dataset, Image, load_from_disk, load_dataset
from PIL import Image as PILImage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
from multiprocessing import Pool
import datasets
# import warnings
# warnings.filterwarnings("error")

basepath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack/"
targetpath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/"

def pad_zeros(number, length):
	number_string = str(number)
	while len(number_string) < length:
		number_string = "0" + number_string
	return number_string

def extract_folder_name(string):
    path = os.path.normpath(string)
    head, tail = os.path.split(path)
    return head

def add_metadata(example):
    item = example['path']
    filename = item.split("/")[-1]
    destpath = extract_folder_name(item)
    id = item.split("/")[-1].split(".")[0]
    txt = destpath + "/" + id + ".txt"
    
    with open(txt, "r", encoding="utf-8") as f:
        text = f.read()
        example['additional_feature'] = text
    
    image = PILImage.open(item)
    example['image'] = image
    return example

def test(example):
    image = example['image']

def process(i):
    print("begin processing", i)
    foderstring = pad_zeros(i, 3)
    destpath = basepath + "data-" + foderstring + "/"

    #try loading
    originlist = list(glob.glob(destpath + "*.jpg"))
    # imglist = []
    # for imgfile in originlist:
    #     try:
    #         img = Image(imgfile) #PILImage.open(imgfile)
    #         imglist.append(imgfile)
    #     except:
    #         print("!!!!!! PILImage fail", imgfile)

    #cast
    ds = Dataset.from_dict({'path': originlist})
    ds = ds.map(add_metadata, num_proc=128)
    ds = ds.remove_columns('path')
    ds = ds.map(test, num_proc=128)
    print(ds[0])

    #ds = ds.cast_column("image", Image())
    outputname = targetpath + "data-" + foderstring + ".hf"
    ds.save_to_disk(outputname)
    
    #examine
    dataset1 = load_from_disk(outputname)
    ds1 = datasets.DatasetDict({"train": dataset1})
    try:
        ds1 = ds1.map(test, num_proc=128)
        print("passed examine, all image ok", i)
    except Exception as e:
        print(e)
        print("!!!!! Something wrong with ", i)


process(2)
#pool = Pool(1)
#pool.map(process, range(2, 3)) 