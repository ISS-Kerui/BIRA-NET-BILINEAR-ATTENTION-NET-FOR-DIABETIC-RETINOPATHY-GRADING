import os
import shutil
train_txt = '/Users/zkr/Desktop/TrainSet.txt'
valid_txt = '/Users/zkr/Desktop/ValidSet.txt'
data_dir = '/Users/zkr/Desktop/dataset/kaggle_diabetic/raw_data'
train_dir = '/Users/zkr/Desktop/dataset/diabetic_classified/train'
valid_dir = '/Users/zkr/Desktop/dataset/diabetic_classified/valid'
with open(train_txt,'r+') as f:
	lines = f.readlines()
	for line in lines:
		if line != '\r\n':
			lis = line.split(' ')
			name = lis[0]
			clss = lis[1].strip()
			shutil.copyfile(os.path.join(data_dir,name+'.jpeg'),os.path.join(train_dir+'/'+clss,name+'.jpeg'))

with open(valid_txt,'r+') as f:
	lines = f.readlines()
	for line in lines:
		if line != '\r\n':
			lis = line.split(' ')
			name = lis[0]
			clss = lis[1].strip()
			shutil.copyfile(os.path.join(data_dir,name+'.jpeg'),os.path.join(valid_dir+'/'+clss,name+'.jpeg'))