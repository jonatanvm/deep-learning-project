import glob
import os
import zipfile
from shutil import rmtree, move
from urllib.request import urlretrieve

# Remove old files
try:
    rmtree("./tiny-imagenet-200")
except Exception:
    pass
try:
    os.remove('./wnids.txt')
except Exception:
    pass
try:
    os.remove('./words.txt')
except Exception:
    pass

print("Downloading tiny-imagenet-200.zip (~240MB)")
urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip", "tiny-imagenet.zip")


print("Unzipping")
zip_ref = zipfile.ZipFile("./tiny-imagenet.zip", 'r')
zip_ref.extractall("./")
zip_ref.close()
rmtree("./tiny-imagenet-200/test")

print("Formatting")
target_folder = './tiny-imagenet-200/val/'
test_folder = './tiny-imagenet-200/test/'
train_folder = './tiny-imagenet-200/train/'
annotations = target_folder + 'val_annotations.txt'
os.mkdir(test_folder)
n_models = len(glob.glob('./models/*'))
if n_models == 0:
    os.mkdir('./models/')

with open('./tiny-imagenet-200/wnids.txt', 'r') as f:
    for class_name in f.readlines():
        os.mkdir(target_folder + class_name.rstrip('\n'))
        os.mkdir(test_folder + class_name.rstrip('\n'))

move('./tiny-imagenet-200/wnids.txt', './wnids.txt')
move('./tiny-imagenet-200/words.txt', './words.txt')

print("Formatting training data.")
paths = glob.glob(train_folder + '*')
for path in paths:
    class_name = path.split('/')[-1]
    os.remove(path + '/' + class_name + "_boxes.txt")
    image_paths = glob.glob(path + '/images/*')
    for image_path in image_paths:
        image_name = image_path.split('/')[-1]
        move(image_path, path + '/' + image_name)
    os.rmdir(path + '/images/')

print("Formatting test and validation data.")
with open(annotations, 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        class_name = split_line[1]
        file_name = split_line[0]
        n_files = len(glob.glob(target_folder + class_name + '/*'))
        if n_files < 25:
            destination = target_folder + class_name + '/' + file_name
        else:
            destination = test_folder + class_name + '/' + file_name
        move('./tiny-imagenet-200/val/images/' + file_name, destination)

os.rmdir(target_folder + 'images')
os.remove(annotations)
os.remove('tiny-imagenet.zip')
print("Done!")
