from collections import defaultdict
import torch

def evaluate_model_per_class(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_names = val_loader.dataset.classes

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, pred = output.max(1)  # get the index of the max log-probability

            for t, p in zip(target.view(-1), pred.view(-1)):
                if t == p:
                    class_correct[class_names[t.item()]] += 1
                class_total[class_names[t.item()]] += 1

            correct += pred.eq(target).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Overall Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')

    for classname in class_names:
        if class_total[classname] > 0:
            class_accuracy = 100. * class_correct[classname] / class_total[classname]
            print(f'Accuracy of {classname} : {class_accuracy:.2f}%')
        else:
            print(f'Accuracy of {classname} : N/A (no samples)')