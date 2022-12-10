# Image Classification Project

Classfication of Mask Wearing Status

- Boostcamp AI Tech
- Project Period: 2022.02.21 - 2022.03.04
- Team CV-09

## Purpose of the Project
Determining an image taken with a camera whether a mask is worn on a person's face<br>

## Overview of the Project
[Wrap Up Report(KOR)](https://yehyunsuh.notion.site/Day35-2022-03-04-2ece3d63496047208789080734267189)<br>

## Dataset
### Training Data
Asian in the age of 20s ~ 70s
- Total number of people: 4500 (60% of the total dataset)
- Picture taken per person: 7<br>
(wearing: 5, wearing incorrectly: 1, not wearing: 1)
- Shape of image: (3,384,512)
- Example:   
<img src="https://user-images.githubusercontent.com/73840274/159110899-9130d774-a65b-4cee-a549-c6209764a244.png" height="200"/><img src="https://user-images.githubusercontent.com/73840274/159110908-6ab7f52d-04a0-4dbb-b68c-68f93b95d29d.jpeg" height="200"/><img src="https://user-images.githubusercontent.com/73840274/159110909-f74ee9ee-2a2f-449d-bd34-720a8e0bdd46.jpeg" height="200"/>   
```---- wearing -- wearing incorrectly -- not wearing```

### Class Description
<img src="https://user-images.githubusercontent.com/73840274/157795574-c75b443d-be6c-4087-925b-61bf59271e0f.png" height="350"/>

### Test Data
- rest 40% of the total dataset

## Final Score
- f1 score: 0.7318   
- accuracy: 79.2381%

### Methods
- Model: Ensemble of results from various models   
(efficientnetb4 x 2, efficientnetb6, resnet18, resnet152, vit_small_patch16_384, vgg19, swin_base_patch4_window7_224)
- Optimizer: Adam
- Loss: Focal Loss, F1 Loss
- Scheduler: stepLR
- Graph of `swin_base_patch4_window7_224`   
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/85d46705-d32d-4a83-890f-5a081fad9bd0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220319%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220319T071427Z&X-Amz-Expires=86400&X-Amz-Signature=469a681af07c12e8baf76f8711ed613c849b4ea5607921c6702c1e3c68c16fe5&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" height="350"/><br>
gray graph: focal loss, oversampling, custom augmentation, lr=5e-4 (`acc: 74.5%, f1 score: 0.694`)   
orange graph: focal loss, oversampling, custom augmentation, lr=1e-4    
blue graph: focal loss, lr=5e-4 (`acc: 73.0%, f1 score: 0.671`)   

## 🛠 Development Tool
OS: Ubuntu   
IDE: Visual Studio Code   
GPU: Tesla V100   

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Downloading
- `git clone https://github.com/justbeaver97/level1-image-classification-level1-cv-09.git`
- `cd level1-image-classification-level1-cv-09/서예현_T3105/`

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

## Developers : Team CV-09
- [x] [서예현_T3105](https://github.com/yehyunsuh)<br>
- [x] [김영운_T3040](https://github.com/Cronople)<br>
- [x] [박기련_T3082](https://github.com/parkgr95)<br>
- [x] [박민수_T3087](https://github.com/mata1139)<br>
- [x] [송민기_T3112](https://github.com/alsrl8)<br>
