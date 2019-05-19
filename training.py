import torch


def compute_accuracy(net, device, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(net, device, n_epochs, optimizer, criterion, trainloader, valloader):
    net.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = 200  # mini-batches
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # Transfer to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i % print_every) == (print_every - 1):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0

        # Print accuracy after every epoch
        accuracy = compute_accuracy(net, device, valloader)
        print('Accuracy of the network on the test images: %.3f' % accuracy)

    print('Finished Training')
