import os 
import shutil
import tarfile
import concurrent.futures
import math

def pad_zeros(number, length):
	number_string = str(number)
	while len(number_string) < length:
		number_string = "0" + number_string
	return number_string
    
originpath = "/nfs/users/zhangsan/datasets/laion-high-resolution-output/"
basepath = "/pfs/sshare/app/zhangsan/laion-high-resolution-unpack/"
for i in range(0, 10137):
	foldernum = math.floor(i/100.0)
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
