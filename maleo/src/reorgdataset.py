# coding=utf-8

import json
import os
import shutil
import tarfile
import math
import logging

logger = logging.getLogger(__name__)


def pad_zeros(number, length):
    number_string = str(number)
    while len(number_string) < length:
        number_string = "0" + number_string
    return number_string


def rebuild_data_0():
    originpath = "/nfs/users/zhangsan/datasets/laion-high-resolution-output/"
    basepath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack/"
    for i in range(0, 10137):
        foldernum = math.floor(i / 100.0)
        foderstring = pad_zeros(foldernum, 3)
        destpath = basepath + "data-" + foderstring + "/"

        if not os.path.exists(destpath):
            os.mkdir(destpath)

        tarstring = pad_zeros(i, 5)
        originfile = originpath + tarstring + ".tar"

        shutil.copy(originfile, destpath)

        destfile = destpath + tarstring + ".tar"

        # tar = tarfile.open(destfile)
        # tar.extractall(destpath)
        # tar.close()
        print(i)


def rebuild_data(srcpath, dstpath, max_count=10000, move_file=True):
    #
    dstindex = 0
    curdst = os.path.join(dstpath, "data-{:0>6}".format(dstindex))
    if not os.path.exists(curdst):
        os.makedirs(curdst)

    logger.info(f"start to write {curdst}")

    # open label file
    metafile = os.path.join(curdst, "metadata.jsonl")
    mfp = open(metafile, "w")
    count = 0
    skipped = 0

    for dir, _, files in os.walk(srcpath):
        for fn in sorted(files):
            if fn[-3:] not in ['jpg', 'png']:
                continue

            skip = False
            for extname in ['txt', 'json']:
                basename = fn[:-3] + extname
                srcfile = os.path.join(dir, basename)
                if not os.path.exists(srcfile):
                    skip = True
                    break

            if skip:
                skipped += 1
                continue

            if count % max_count == max_count - 1:
                mfp.close()
                dstindex += 1
                curdst = os.path.join(dstpath, "data-{:0>6}".format(dstindex))
                os.mkdir(curdst)
                mfp = open(os.path.join(curdst, "metadata.jsonl"), "w")
                logger.info(f"start to write {curdst}")

            # copy
            if move_file:
                shutil.move(os.path.join(dir, fn), os.path.join(curdst, fn))
            else:
                shutil.copy(os.path.join(dir, fn), os.path.join(curdst, fn))
            for extname in ['txt', 'json']:
                basename = fn[:-3] + extname
                srcfile = os.path.join(dir, basename)
                if move_file:
                    shutil.move(srcfile, os.path.join(curdst, basename))
                else:
                    shutil.copy(srcfile, os.path.join(curdst, basename))

            # write to meta
            with open(os.path.join(curdst, fn[:-3] + "txt"), "r", encoding="utf-8") as fp:
                text = fp.read()
                data = {"file_name": fn, "caption": text}
                json.dump(data, mfp, ensure_ascii=False)
                mfp.write("\n")

            count += 1
    mfp.close()

    logger.info(f"total processed: {count},  skipped: {skipped}")


if __name__ == "__main__":
    format = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=format)

    srcpath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack/"
    dstpath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack2/"

    rebuild_data(srcpath, dstpath, max_count=10000, move_file=True)
