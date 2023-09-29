
import logging
import os
import random
import json
import tarfile
# import functools
import torch
from PIL import Image
# from streaming import Stream, StreamingDataset
# from torch.utils.data import DataLoader, IterableDataset, Dataset
from datasets import DatasetDict, Dataset, IterableDataset
# from pyarrow import parquet as pq

logger = logging.getLogger('laion_dataset')


class LocalLaionShard:
    def __init__(self, path: str, name: str) -> None:
        self.path = path
        self.name = name
        self.samples = []
        self.members = None
        self.fp = None

    def size(self):
        # sample count
        return len(self.samples)
    
    def shuffle(self, seed=None):
        if seed:
            random.seed(seed)
        random.shuffle(self.samples)

    def get_sample(self, idx):
        s = self.samples[idx]
        # s is tuple: 
        # 0: basename,  1: index of tarinfo for jpg,   2: index of tarinfo for json,   3: index of tarinfo for txtfile 
        img = Image.open(self.fp.extractfile(self.members[s[1]]))
        meta = json.load(self.fp.extractfile(self.members[s[2]]))

        # 这里暂不处理tokenizer,  img.resize
        return {'image': img, 'caption': meta['caption']}

    def load(self):
        tarname = os.path.join(self.path, self.name + ".tar")
        # pqname = os.path.join(self.path, self.name + ".parquet")
        # table = pq.read_table(pqname).to_pandas()

        try:
            self.fp = tarfile.open(tarname, "r")
            self.members = self.fp.getmembers()
        except Exception as e:
            logger.error("unable to open %s: %s", tarname, e)
            return False

        # 构建索引：
        # group by name
        sample_index = {}
        name_index = {}
        for i, tarinfo in enumerate(self.members):
            # .jpg/.json/.txt
            name_index[tarinfo.name] = i
            parts = tarinfo.name.split('.')
            if len(parts) < 2:
                continue
            if parts[-1] not in {'jpg', 'json', 'txt'}:
                continue
            basename = '.'.join(parts[0:-1])
            if basename not in sample_index:
                sample_index[basename] = {parts[-1]: i}
            else:
                sample_index[basename][parts[-1]] = i
        
        # 构建数据集: 
        # 0: basename,  1: index of tarinfo for jpg,   2: index of tarinfo for json,   3: index of tarinfo for txtfile 
        samples = []
        for k, v in sample_index.items():
            if len(v) == 3:
                samples.append((k, v['jpg'], v['json'], v['txt']))

        self.samples = samples
        return True    

class LocalLaionDataSet(IterableDataset):
    def __init__(self, file_paths, rank=0, num_work=0) -> None:
        # file_paths: [str, list],  文件路径
        # rank: work编号
        # num_work: 总work数
        super().__init__()
        self.file_paths = file_paths
        self.rank = rank
        self.num_work = num_work
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        self.shard_list = []
        self.load()
    
    def __iter__(self):
        for shard_idx, shard in enumerate(self.shard_list):
            for sub_idx in range(len(shard.samples)):
                yield self._get_sample(shard_idx, sub_idx)
        return None

    def __getitem__(self, index):
        # todo convert index to shard_idx, sub_idx
        shard_idx, sub_idx = 0, 0
        return self._get_sample(shard_idx, sub_idx)
    
    def _get_sample(self, shard_idx, sub_idx):
        sample = self.shard_list[shard_idx].get_sample(sub_idx)
        # todo:
        # 处理图像：resize,  mode RGB...
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        sample['image'] = img

        # 处理文本：tokenizer ... 
        return sample

    def shuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.shard_list)
        self._build_index()

        for shard in self.shard_list:
            shard.shuffle()

    def load(self):
        for path in self.file_paths:
            for dir, _, files in os.walk(path):
                logger.info(f"{dir}: {len(files)}")
                for fn in sorted(files):
                    if not fn.endswith(".tar"):
                        continue
                    shard = self._load_shard(dir, fn[0:-4])
                    if shard:
                        self.shard_list.append(shard)
                    if len(self.shard_list) == 10:
                        break
        
        # 构建索引
        self._build_index()

        logger.info(f"total shard: {len(self.shard_list)}, count: {sum(self.shard_size)}")
        
    def _load_shard(self, path: str, name: str):
        shard = LocalLaionShard(path, name)
        if shard.load():
            logger.info(f"{name}: {shard.size()}")
            return shard
        return None

    def _build_index(self):
        self.shard_size = [shard.size() for shard in self.shard_list]

def load_laion_dataset(path, num_work=0, rank=0):
    dataset = LocalLaionDataSet(path, rank, num_work)
    d = {'train': dataset}
    return DatasetDict(d)

def test_enumerate(dataset):
    for i, s in enumerate(dataset):
        if i % 10000 == 0:
            logger.info(s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s: %(message)s")

    # path = "/ML-A100/data/multimodal_data/laion2B-en/laion2B-en/"
    path = "/ML-A100/data/multimodal_data/laion2B-en/laion2B-en/part-00048-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet.slice.3"

    path = "/ML-A100/sshare-app/zhangsan/laion-high-resolution-output"

    # path = "/nfs/users/zhangsan/datasets/laion-high-resolution-output"
    ds = load_laion_dataset(path)
    ds['train'].shuffle(17)

    test_enumerate(ds['train'])
