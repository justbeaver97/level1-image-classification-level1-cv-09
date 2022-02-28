import wandb
import PIL.Image
from typing import Optional

class MyLogger():
    def __init__(self) -> None:
        wandb.init(project="my project", entity="songmingi")
        
    def log(self, data):
        wandb.log(data)

    def log_images(self, key, images, caption:Optional[str] = None):
        images = wandb.Image(images, caption)
        wandb.log({key: images})
        
    
    def close(self):
        wandb.finish()


if __name__ == '__main__':

    # test code
    logger = MyLogger()
    logger.log({'test': 1})
    logger.log({'test': 2})
    img = PIL.Image.open('/opt/ml/code/smg/data/test_image.png')
    logger.log_images("test", img)  
    logger.close()
    