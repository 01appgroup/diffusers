import os 
import shutil
import tarfile
import concurrent.futures
import math

def extract_tar_file(tar_file_path, output_folder):
	with tarfile.open(tar_file_path) as tar:
		tar.extractall(output_folder)

def extract_tar_files_in_parallel(tar_file_paths, output_folder, num_threads=128):
	with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
		executor.map(extract_tar_file, tar_file_paths, [output_folder] * len(tar_file_paths))
		
def pad_zeros(number, length):
	number_string = str(number)
	while len(number_string) < length:
		number_string = "0" + number_string
	return number_string

def list_tar_files(folder_path):
    tar_file_paths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tar"):
            tar_file_paths.append(os.path.join(folder_path, file_name))
    return tar_file_paths

originpath = "/nfs/users/zhangsan/datasets/laion-high-resolution-output/"
basepath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack/"

for i in range(14, 102):
	foderstring = pad_zeros(i, 3)
	destpath = basepath + "data-" + foderstring + "/"
	tarfiles = list_tar_files(destpath)
	extract_tar_files_in_parallel(tarfiles, destpath)
	print(i)