import torch

def train(model, device, train_loader, criterion, optimizer, epoch, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f} Avg Acc: {100. * correct / total:.2f}%')

    if scheduler:
        scheduler.step()
    print(f'End of Epoch {epoch}: Average Loss: {running_loss / len(train_loader):.4f}, Average Accuracy: {100. * correct / total:.2f}%')