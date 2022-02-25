import wandb

class MyLogger():
    def __init__(self) -> None:
        wandb.init(project="my project", entity="songmingi")
        
    def log(self, data):
        wandb.log(data)
    
    def close(self):
        wandb.finish()


if __name__ == '__main__':

    # test code
    logger = MyLogger()
    logger.log({'test': 1})
    logger.log({'test': 2})
    logger.close()
    