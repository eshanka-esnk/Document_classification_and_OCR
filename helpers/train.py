import torch
from torch.utils.data.dataloader import DataLoader

def accuracy(outputs: torch.tensor, labels: torch.tensor, threshold: float=0.5) -> torch.tensor:

    preds = (outputs >= threshold).float()
    correct = (preds == labels).float().mean()
    return correct

@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: str):

    model.eval()
    outputs = [model.validation_step(images.to(device), labels.to(device)) for images, labels in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs: int, learning_rate: float, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, save_best_model=None, opt_func: torch.optim=torch.optim.Adam, device: str="cpu"):
    
    model.to(device)
    history = []
    optimizer = opt_func(model.parameters(), learning_rate)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for images, labels in train_loader:
            loss = model.training_step(images.to(device), labels.to(device))
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        if save_best_model != None:
            save_best_model(result['val_loss'], epoch, model, optimizer)
        history.append(result)
    
    return model, optimizer, history