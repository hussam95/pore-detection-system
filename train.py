from dataset import CellImageDataset
from model import PoreNet, data_transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def main():
    # the rest of your code goes here
    working_dir = os.getcwd()
    # Load the dataset
    dataset = CellImageDataset(root_dir=working_dir + '\\ct-images', csv_file=working_dir + '\\pore_data.csv', transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Create the model and optimizer
    model = PoreNet()
    
    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Push model to cuda
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # Print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished training')

    # Save the model
    torch.save(model.state_dict(), 'porenet.pth')

if __name__ == '__main__':
    main()
