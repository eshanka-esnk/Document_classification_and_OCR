import torch
from helpers.train import accuracy

class CNNImageClassification(torch.nn.Module):

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        self.network = torch.nn.Sequential(
            
            torch.nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
        
            torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            
            torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            
            torch.nn.Flatten(),
            torch.nn.Linear(200704,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,6)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
    def training_step(self, images, labels):
        out = self(images)
        loss = self.criterion(out, labels)
        return loss
    
    def validation_step(self, images, labels):
        out = self(images)
        loss = self.criterion(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("[INFO]: Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        