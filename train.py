import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import MLPClassifier

def train_model(X_train, X_val, y_train, y_val, params, utils):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True)

    model = MLPClassifier(input_dim=X_train.shape[1],
                          hide1_num=params['hide1_num'],
                          hide2_num=params['hide2_num'],
                          dropout1=params['dropout1'],
                          dropout2=params['dropout2'],
                          output_dim=len(torch.unique(y_train)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    losses = []
    accuracies = []
    validation_losses = []
    validation_accuracy = []

    min_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(params['num_epochs']):
        model.train()
        total_loss = 0
        total_accuracy = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).float().sum()

            total_accuracy += correct / targets.shape[0]
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        average_accuracy = total_accuracy / len(train_loader) 
        losses.append(average_loss)
        accuracies.append(average_accuracy.to('cpu'))

        model.eval()
        with torch.no_grad(): 
            val_total_loss = 0
            val_total_accuracy = 0
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == targets).float().sum()

                val_total_accuracy += correct / targets.shape[0]
                val_total_loss += loss.item()
            
            val_loss = val_total_loss / len(val_loader)
            val_accuracy = val_total_accuracy / len(val_loader) 
            validation_losses.append(val_loss)
            validation_accuracy.append(val_accuracy.to('cpu'))

        if val_loss < min_val_loss: 
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == params['early_stop']: 
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    paramsStr =  "\n- ".join([""]+[f"{k}:\t{v}" for k,v in params.items()])
    log = f'''<hr>

# ID {params['id']}

- Params {paramsStr}
- Epoch {epoch+1}
- Training Loss: {round(average_loss,3)}, Training Accuracy: {round(float(average_accuracy),3)}
- Validation Loss: {round(val_loss,3)}, Validation Accuracy: {round(float(val_accuracy),3)}

'''

    utils.log(log)

    utils.save_loss_acc_plot(losses, accuracies, validation_losses, validation_accuracy, params['id'])

    return model, losses, accuracies, validation_losses, validation_accuracy
