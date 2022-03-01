import os
import csv
import PIL.Image
import numpy as np
from enum import Enum
from torch.utils.data import Dataset


class DataType(Enum):
    MASK = 'mask'
    GENDER = 'gender'
    AGE = 'age'


class DatasetMode(Enum):
    TRAIN = 'train'
    VAL = 'val'
    

def split_data(data_path:str='/opt/ml/input/data/train', \
                csv_path:str='/opt/ml/code/smg/cascade/', \
                val_ratio:float=0.0):

    # validation
    if val_ratio < 0 or val_ratio >= 1.0:
        raise ValueError("val_ratio must be greater than 0 and less than 1.0")
    
    dirs_path = os.path.join(data_path, 'images')
    dirs = next(os.walk(dirs_path))[1]
    data = []
    for dir in dirs:
        images_path = os.path.join(dirs_path, dir)
        images = next(os.walk(images_path))[2]
        images = [image for image in images if not image.startswith('._')]
        for image in images:
            data.append([dir, image])
    
    indexes = np.random.permutation(len(data)).tolist()
    val_num = int(len(data) * val_ratio)

    f = open(os.path.join(csv_path, 'val.csv'), 'w')
    wr = csv.writer(f)
    for i in range(val_num):
        wr.writerow(data[indexes[i]])
    f.close()

    f = open(os.path.join(csv_path, 'train.csv'), 'w')
    wr = csv.writer(f)
    for i in range(val_num, len(data)):
        wr.writerow(data[indexes[i]])
    f.close()

    print(f'train dataset num: {len(data)-val_num}')
    print(f'validation dataset num: {val_num}')

    
class CascadeDataset(Dataset):

    def __init__(self, data_path:str='/opt/ml/input/data/train/images', \
                csv_path:str='/opt/ml/code/smg/cascade/', \
                transforms=None, mode:DatasetMode=None, \
                dataType:DataType=None) -> None:
        
        # validation
        if mode is None:
            raise RuntimeError("mode cannot be None")
        elif dataType is None:
            raise RuntimeError("dataType cannot be None")
        elif type(dataType) is not DataType:
            raise TypeError("dataType must be in DataType Enum")
        
        self.data_path = data_path
        self.csv_path = os.path.join(csv_path, mode.value + '.csv')
        self.transforms = transforms
        self.mode = mode
        self.dataType = dataType
        self.data = self.__read_csv__()
        self.__put_label()

    
    def __read_csv__(self):
        f = open(self.csv_path, 'r')
        rd = csv.reader(f)
        images = [row for row in rd]
        f.close()
        return images


    def __put_label(self):
        csv_path = os.path.join('/opt/ml/code/smg/cascade', self.dataType.value + '_' + self.mode.value + '.csv')
        f = open(csv_path, 'w')
        wr = csv.writer(f)

        label = ''
        for i in range(len(self.data)):
            dir, img = self.data[i]
            idx, gender, race, age = dir.split('_')
            if self.dataType is DataType.MASK:
                if img.startswith('mask'):
                    label = 0  # Wear
                elif img.startswith('normal'):
                    label = 1  # Not Wear
                elif img.startswith('incorrect_mask'):
                    label = 2  # Incorrect
            elif self.dataType is DataType.GENDER:
                if gender == 'male':
                    label = 0  # Male
                elif gender == 'female':
                    label = 1  # Female
            elif self.dataType is DataType.AGE:
                label = int(age) // 30
            
            if label == '':
                raise RuntimeError("labeling is failed")
        
            self.data[i].append(label)
            wr.writerow([dir, img, label])
        f.close()


    def __len__(self):
        return len(self.data)

    
    def __repr__(self):
        return "=" * 25 + "\n" + \
                f"Custom {self.dataType.value} Dataset for {self.mode.value}\n" + \
                f"data num: {len(self)}\n" + \
                f"transforms: {self.transforms}\n" + \
                f"dataType: {self.dataType.value}\n" + \
                f"mode: {self.mode.value}\n" + \
                "=" * 25


    def __getitem__(self, idx):
        dir, img, label = self.data[idx]
        img_path = os.path.join(self.data_path, dir, img)
        img = PIL.Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img, label
            

# test code
if __name__ == '__main__':

    print('test started')

    split_data(val_ratio=0.5)

    mask_train_dataset = CascadeDataset(mode=DatasetMode.TRAIN, dataType=DataType.MASK)
    mask_val_dataset = CascadeDataset(mode=DatasetMode.VAL, dataType=DataType.MASK)
    gender_train_dataset = CascadeDataset(mode=DatasetMode.TRAIN, dataType=DataType.GENDER)
    gender_val_dataset = CascadeDataset(mode=DatasetMode.VAL, dataType=DataType.GENDER)
    age_train_dataset = CascadeDataset(mode=DatasetMode.TRAIN, dataType=DataType.AGE)
    age_val_dataset = CascadeDataset(mode=DatasetMode.VAL, dataType=DataType.AGE)
    
    print('test finished')