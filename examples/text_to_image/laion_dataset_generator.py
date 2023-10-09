# coding=utf-8

# generator of datasets.Dataset.from_generator, for laion dataset in tar files
# examples:
# to load from data_files. (support num_proc)
#     data_dir = "/ML-A100/sshare-app/zhangsan/laion-high-resolution-output"
#     data_files = [os.path.join(data_dir, "{:0>5}.tar".format(fn)) for fn in range(0, 32)]
#     dataset = datasets.Dataset.from_generator(laion_dataset_generator,
#                                               gen_kwargs={"data_files": data_files},
#                                               cache_dir="output/cache/",
#                                               num_proc=32,
#                                               config_name="laion_high_resolution", version="0.0.0")
#
# to load from data_dir. (do not support num_proc)
#     data_dir = "/ML-A100/sshare-app/zhangsan/laion-high-resolution-output"
#     dataset = datasets.Dataset.from_generator(laion_dataset_generator,
#                                               gen_kwargs={"data_dir": data_dir},
#                                               cache_dir="output/cache/",
#                                               num_proc=32,
#                                               config_name="laion_high_resolution", version="0.0.0")


import logging
import json
import os
import tarfile
from PIL import Image


logger = logging.getLogger(__name__)


def build_index(members: list):
    sample_index = {}
    name_index = {}
    for i, tarinfo in enumerate(members):
        # .jpg/.json/.txt
        name_index[tarinfo.name] = i
        parts = tarinfo.name.split('.')
        if len(parts) < 2:
            continue
        if parts[-1] not in ('jpg', 'json', 'txt'):
            continue
        basename = '.'.join(parts[0:-1])
        if basename not in sample_index:
            sample_index[basename] = {parts[-1]: i}
        else:
            sample_index[basename][parts[-1]] = i
    return sample_index


def generate_sample_from_tarfile(tarname: str, min_image_size: int = None, **kwargs):
    with tarfile.open(tarname, "r") as fp:
        members = fp.getmembers()
        index = build_index(members)

        # logger.info(f'{tarname}: total {len(members)} files in tar, {len(index)} samples.')

        for k, v in index.items():
            if len(v) != 3:
                continue
            try:
                image = Image.open(fp.extractfile(members[v['jpg']]))
                # skip small-image
                if min_image_size:
                    if image.width < min_image_size or image.height < min_image_size:
                        continue

                image.load()

                text = fp.extractfile(members[v['txt']]).read().decode()

            except Exception as e:
                # discard error item
                logger.error(f"Error for {k}: {e}")
                continue
            yield k, {
                # "caption": meta.get('caption', ''),
                "caption": text,
                "image": image
            }
        fp.close()


def generate_sample_from_dir(data_dir: str, max_file_num=None, **kwargs):
    count = 0
    for dir, _, files in os.walk(data_dir):
        # logger.info(f"{dir}: {len(files)} files")
        for fn in files:
            count += 1
            if max_file_num and count > max_file_num:
                break

            if not fn.endswith('.tar'):
                continue
            tarname = os.path.join(dir, fn)
            for k, item in generate_sample_from_tarfile(tarname):
                yield k, item
        break
    pass


def laion_dataset_generator(data_dir: str = None, data_files: list = None, **kwarg):
    if data_files:
        for fn in data_files:
            for k, item in generate_sample_from_tarfile(fn, **kwarg):
                yield item
    elif data_dir:
        for k, item in generate_sample_from_dir(data_dir, **kwarg):
            yield item
