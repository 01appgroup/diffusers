import os
import json

def pad_zeros(number, length):
	number_string = str(number)
	while len(number_string) < length:
		number_string = "0" + number_string
	return number_string

def list_jpg_files(folder_path):
    jpg_file_paths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):
            jpg_file_paths.append(os.path.join(folder_path, file_name))
    return jpg_file_paths

basepath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack/"

for i in range(49, 101):
    foderstring = pad_zeros(i, 3)
    destpath = basepath + "data-" + foderstring + "/"
    jpgfiles = list_jpg_files(destpath)
    metafile = destpath + "metadata.jsonl"

    with open(metafile, "w") as mf:
        for item in jpgfiles:
            filename = item.split("/")[-1]
            id = item.split("/")[-1].split(".")[0]
            txt = destpath + id + ".txt"
            
            with open(txt, "r", encoding="utf-8") as f:
                text = f.read()
        
            data = {"file_name": filename, "additional_feature": text}
            json.dump(data, mf, ensure_ascii=False)
            mf.write("\n")
    print(i)