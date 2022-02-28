import os
import csv
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class TrainDataset(Dataset):
    """
    Description
    ===
    custom dataset by songmingi
    
    Parameters
    ===
    root: parent directory of images path
    transforms: data transforms
    val_ratio: ratio of validation set to total data set(0.0 <= x < 1)
    """

    def __init__(self, root: str = '/opt/ml/input/data/train', \
                transforms = None, val_ratio = 0.0):
        self.path = os.path.join(root, 'images')
        self.transforms = transforms
        self.val_ratio = val_ratio
        self.LABEL_NUM = 18
        self.classes = {'Mask': {'Wear': 0, 'Incorrect':1, 'Not Wear':2},
                        'Gender': {'Male':0, 'Female':1}, 
                        'Age': lambda x:x//30}
        self.dirs = next(os.walk(self.path))[1]
        self.images = {label:[] for label in range(self.LABEL_NUM)}
        self.nums = {label:0 for label in range(self.LABEL_NUM)}

        # variables validation
        if not 0.0 <= self.val_ratio < 1:
            raise ValueError("val_ratio must be greater than or equal to 0 and less than 1")

        self.__read_images()
        self.train_images, self.val_images = self.__split_validation_set()
        self.__write_csv('train.csv', self.train_images)
        self.__write_csv('val.csv', self.val_images)


    def __read_images(self):
        """
        read images and get their labels. calculate number of image set
        """
        for dir in self.dirs:
            images_path = os.path.join(self.path, dir)
            images = next(os.walk(images_path))[2]
            for image in images:
                if image.startswith('._'):
                    continue
                label = self.__get_label_with_filename(dir, image)
                self.images[label].append([dir, image])
                self.nums[label] += 1
                
    
    def __split_validation_set(self):
        """
        split image set to train set and validation set
        """
        train_images = []
        val_images = []
        for label in range(self.LABEL_NUM):
            val_num = int(self.nums[label] * self.val_ratio)
            indicies = np.random.permutation(len(self.images[label])).tolist()
            i = 0
            while i < val_num:
                val_images.append(self.images[label][indicies[i]] + [label])
                i += 1
            while i < self.nums[label]:
                train_images.append(self.images[label][indicies[i]] + [label])
                i += 1
        return train_images, val_images

        
    def __get_label_with_filename(self, dirname: str, filename: str) -> int:
        """
        get label with directory name and file name
        """
        idx, gender, race, age = dirname.split('_')
        mask = ''
        if filename.startswith('normal'):
            mask = 'Not Wear'
        elif filename.startswith('incorrect_mask'):
            mask = 'Incorrect'
        else:
            mask = 'Wear'

        if gender == 'male':
            gender = 'Male'
        elif gender == 'female':
            gender = 'Female'

        label_num = self.classes['Mask'][mask] * 6 + self.classes['Gender'][gender] * 3 + self.classes['Age'](int(age))
        return label_num

    
    def __write_csv(self, csv_path, image_list):
        """
        write csv file with image data on specific path
        """
        f = open(csv_path, 'w')
        wr = csv.writer(f)

        wr.writerow(('label', 'dir', 'file'))
        for dir, file, label in image_list:
            wr.writerow((dir, file, label))
        
        f.close()
                
    
    def __len__(self):
        return len(self.train_images)


    def __repr__(self):
        return '=' * 25 + '\nCustom Dataset\n' + \
                f'total data num: {len(self)}\n' + \
                f'train data num: {sum(self.nums.values())}\n' + \
                f'path: {self.path}\n' + \
                f'transforms: {self.transforms}\n' + \
                f'val_ratio: {self.val_ratio}\n' + \
                '=' * 25

    
    def __getitem__(self, idx):
        dir, image, label = self.train_images[idx]
        img_path = os.path.join(self.path, dir, image)
        img = PIL.Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img, label





class ValidationDataset(Dataset):
    
    def __init__(self, root: str = '/opt/ml/input/data/train', \
                transforms = None):
        self.path = os.path.join(root, 'images')
        self.transforms = transforms
        self.val_images = self.__read_csv('val.csv')


    def __read_csv(self, csv_path):
        """
        read validtion csv file
        """
        val_images = []
        f = open(csv_path, 'r')
        rd = csv.reader(f)

        rd.__next__()
        for row in rd:
            dir, file, label = row
            val_images.append([dir, file, int(label)])
        
        f.close()
        return val_images

    
    def __len__(self):
        return len(self.val_images)


    def __repr__(self) -> str:
        return '=' * 25 + '\nCustom Dataset for validation\n' + \
                f'total data num: {len(self)}\n' + \
                f'path: {self.path}\n' + \
                f'transforms: {self.transforms}\n' + \
                '=' * 25

    
    def __getitem__(self, idx):
        dir, image, label = self.val_images[idx]
        img_path = os.path.join(self.path, dir, image)
        img = PIL.Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img, label
        




if __name__ == '__main__':
    # test code
    try:
        transforms = transforms.Compose([transforms.ToTensor()])
        ds = TrainDataset(transforms=transforms, val_ratio=0.3)
        vs = ValidationDataset(transforms=transforms)
        
        print(ds)
        print(vs)
        
    except ValueError as e:
        print(e)
