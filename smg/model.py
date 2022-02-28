import torch
import torch.cuda
import torch.nn as nn
from torch import optim
from torchvision.models import vgg19
from torch.nn.modules.linear import Linear
from copy import deepcopy
from logger import MyLogger


def get_pretrained_vgg19(class_num: int):
    """
    Parameter
    ===
    class_num: the number of classes

    Return
    ===
    return pretrained vgg19 model.
    a linear layer with out_features of class_num was added
    """
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    model = vgg19(pretrained=True)
    last_out = model.classifier[-1].out_features
    model.classifier.add_module(str(len(model.classifier)), Linear(last_out, class_num))
    return model.to(device)


def train_model(model, dataloader, num_epochs, device=None, criterion=None, optimizer=None):
    best_model = None
    best_acc = 0.0

    if not device:
        device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    if not criterion:
        criterion = nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(f'device: {device}')

    logger = MyLogger()

    for epoch in range(1, num_epochs+1):
        print('=' * 25)
        print(f'Epoch: {epoch}/{num_epochs}')

        running_loss = {'train': 0, 'val': 0}
        running_corrects = {'train': 0, 'val': 0}

        for phase in ['train', 'val']:
            log_cnt = 0
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if log_cnt == 0:  # log images for every 5 training        
                    log_cnt = 5
                    caption = [phase] + labels.tolist()
                    logger.log_images("images", inputs, caption=caption)
                    
                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                running_loss[phase] += loss.item() * inputs.size(0)
                running_corrects[phase] += torch.sum(preds == labels.data)

                log_cnt -= 1
    
        train_loss = running_loss['train'] / len(dataloader['train'])
        train_acc = running_corrects['train'] / len(dataloader['train'])
        val_loss = running_loss['val'] / len(dataloader['val'])
        val_acc = running_corrects['val'] / len(dataloader['val'])

        print(f'train loss: {train_loss}, train acc: {train_acc}')
        print(f'val loss: {val_loss}, val acc: {val_acc}')
        logger.log({'train.loss':train_loss})
        logger.log({'train.acc':train_acc})
        logger.log({'val.loss': val_loss})
        logger.log({'val.acc': val_acc})
        if best_acc < val_acc:
            best_acc = val_acc
            best_model = deepcopy(model.state_dict())
    
    return best_model


def save_model_state(model, path):
    torch.save(model, path)
