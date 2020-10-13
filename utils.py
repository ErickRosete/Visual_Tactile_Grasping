import torch

def save_model(epoch, model, optimizer, scheduler, loss, path):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    }, path)

def eval_model(model, criterion, val_loader):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (data, labels) in enumerate(val_loader):
            outputs = model(data["cam_bef"], data["cam_dur"], data["lgel_bef"], 
                            data["lgel_dur"], data["rgel_bef"], data["rgel_dur"])
            predicted = torch.argmax(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        val_accuracy = correct / total
        val_loss = running_loss / i
    model.train()
    return val_accuracy, val_loss
