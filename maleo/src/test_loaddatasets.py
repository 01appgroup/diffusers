import logging
import os
import datasets
from laion_dataset import laion_dataset_generator

logger = logging.getLogger(__name__)

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
    # print(image)


def test_load():
    dataset0 = datasets.load_from_disk("/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile/data-005.hf")
    ds = datasets.DatasetDict({"train": dataset0})
    # ds = ds.map(test, num_proc=128)  # num_proc for multiprocessing

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


def test_load_onfly():
    data_dir = "/ML-A100/sshare-app/zhangsan/laion-high-resolution-output"

    # logger.info("=====load_dataset from imagefolder")
    # dataset = datasets.load_dataset("imagefolder",
    #                                 cache_dir="output/cache/",
    #                                 data_dir="~/work/laion-high-resolution/unpacked/05133")

    # logger.info(dataset)
    # logger.info(dataset["train"][0])

    logger.info("=====load_dataset by gen")

    data_files = [os.path.join(data_dir, "{:0>5}.tar".format(fn)) for fn in range(0, 32)]
    dataset = datasets.Dataset.from_generator(laion_dataset_generator,
                                              gen_kwargs={"data_files": data_files},
                                              cache_dir="output/cache/",
                                              num_proc=32,
                                              config_name="laion_high_resolution")

    # dataset = datasets.Dataset.from_generator(laion_dataset_generator,
    #                                           gen_kwargs={"data_dir": data_dir, "max_file_num": 32},
    #                                           cache_dir="output/cache/",
    #                                           num_proc=32,
    #                                           config_name="laion_high_resolution")

    logger.info(dataset)
    # logger.info(dataset[0])

    logger.info("load_dataset done")

    dataset.save_to_disk(dataset_path="~/work/laion-high-resolution/data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s: %(message)s")

    test_load_onfly()
