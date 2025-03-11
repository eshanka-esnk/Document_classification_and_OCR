import torch
from helpers.model import CNNImageClassification

class SaveBestModel:

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss,epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\n[INFO]: Best validation loss: {self.best_valid_loss}")
            print(f"\n[INFO]: Saving best model for epoch: {epoch+1}\n")
            print(f"------------------------------------------------------------------------------\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': model.criterion,
                }, 'models/best_model.pth')

def save_model(epochs, model, optimizer):

    print(f"Saving model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': model.criterion,
                }, 'models/final_model.pth')

def load_model(path: str, device: str, verbose: bool=False):
    model = CNNImageClassification(torch.nn.functional.cross_entropy).to(device)
    last_model_cp = torch.load(path, weights_only=False)
    model.load_state_dict(last_model_cp['model_state_dict'])
    if verbose:
        print(model.eval())
    return model

def test_model(model, testloader, device):
    model.eval()
    print('[INFO]: Testing...')
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for image, labels in testloader:
            image = image.to(device)
            labels = labels.to(device)
            predict = model(image)
            max_value, max_indices = torch.max(predict.data, 1)
            label_indices = torch.argmax(labels, dim=1) 
            correct_predictions += (max_indices == label_indices).sum().item()
            total_predictions += labels.size(0)

            for i in range(len(max_indices)):
                predicted_index = max_indices[i].item()
                label = label_indices[i].item()
                if predicted_index == label:
                    print(f"[INFO]: Predicted index: {predicted_index}, Label: {label}, Predicted index matches label")
                else:
                    print(f"[INFO]: Predicted index: {predicted_index}, Label: {label}, Predicted index does not match label")

    accuracy = correct_predictions / total_predictions
    print(f"[INFO]: Accuracy: {accuracy:.4f}")