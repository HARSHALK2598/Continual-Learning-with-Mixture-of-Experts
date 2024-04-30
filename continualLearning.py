import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the saliency map computation function
def compute_saliency_maps(model, inputs, targets):
    model.eval()
    fc1_saliency_maps = []
    fc2_saliency_maps = []
    fc3_saliency_maps = []
    for input, target in zip(inputs, targets):
        input = input.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        output = model(input)
        loss = F.cross_entropy(output, target)
        fc1_weights = list(model.fc1.parameters())[0]
        fc1_grads = torch.autograd.grad(loss, fc1_weights, retain_graph=True)[0]
        fc1_saliency_map = torch.abs(fc1_grads)
        fc1_saliency_maps.append(fc1_saliency_map)
        fc2_weights = list(model.fc2.parameters())[0]
        fc2_grads = torch.autograd.grad(loss, fc2_weights, retain_graph=True)[0]
        fc2_saliency_map = torch.abs(fc2_grads)
        fc2_saliency_maps.append(fc2_saliency_map)
        fc3_weights = list(model.fc3.parameters())[0]
        fc3_grads = torch.autograd.grad(loss, fc3_weights)[0]
        fc3_saliency_map = torch.abs(fc3_grads)
        fc3_saliency_maps.append(fc3_saliency_map)
    fc1_mean_saliency_map = torch.mean(torch.stack(fc1_saliency_maps), dim=0)
    fc2_mean_saliency_map = torch.mean(torch.stack(fc2_saliency_maps), dim=0)
    fc3_mean_saliency_map = torch.mean(torch.stack(fc3_saliency_maps), dim=0)
    model.train()
    return fc1_mean_saliency_map, fc2_mean_saliency_map, fc3_mean_saliency_map

# Define the function to create binary masks
def create_binary_masks(fc1_saliency_map, fc2_saliency_map, fc3_saliency_map, k):
    fc1_flattened_saliency_map = fc1_saliency_map.flatten()
    fc2_flattened_saliency_map = fc2_saliency_map.flatten()
    fc3_flattened_saliency_map = fc3_saliency_map.flatten()
    fc1_sorted_indices = torch.argsort(torch.abs(fc1_flattened_saliency_map), descending=True)
    fc2_sorted_indices = torch.argsort(torch.abs(fc2_flattened_saliency_map), descending=True)
    fc3_sorted_indices = torch.argsort(torch.abs(fc3_flattened_saliency_map), descending=True)
    fc1_top_k_indices = fc1_sorted_indices[:k]
    fc2_top_k_indices = fc2_sorted_indices[:k]
    fc3_top_k_indices = fc3_sorted_indices[:500]

    fc1_binary_mask = torch.zeros_like(fc1_flattened_saliency_map)
    fc1_binary_mask[fc1_top_k_indices] = 1
    fc2_binary_mask = torch.zeros_like(fc2_flattened_saliency_map)
    fc2_binary_mask[fc2_top_k_indices] = 1
    fc3_binary_mask = torch.zeros_like(fc3_flattened_saliency_map)
    fc3_binary_mask[fc3_top_k_indices] = 1
    return fc1_binary_mask.view(500, 784), fc2_binary_mask.view(500, 500), fc3_binary_mask.view(10, 500)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define the tasks
tasks = [
    [(1, 5), (mnist_train.targets == 1) | (mnist_train.targets == 5)],
    [(2, 6), (mnist_train.targets == 2) | (mnist_train.targets == 6)],
    [(3, 7), (mnist_train.targets == 3) | (mnist_train.targets == 7)],
    [(4, 8), (mnist_train.targets == 4) | (mnist_train.targets == 8)],
    [(5, 9), (mnist_train.targets == 5) | (mnist_train.targets == 9)],
    [(0, 2), (mnist_train.targets == 0) | (mnist_train.targets == 2)]
]

# Continual learning loop
model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
alpha = 0.5
threshold = 20.0
test_accuracies = {task[0]: [] for task in tasks}
# Initialize the tensors with zeros
global_fc1_mask = torch.zeros(500, 784).to(device)
global_fc2_mask = torch.zeros(500, 500).to(device)
global_fc3_mask = torch.zeros(10, 500).to(device)
binary_masks = []
prev_inputs, prev_targets = None, None
prevLc = 0
for task_idx, (task_labels, task_indices) in enumerate(tasks):
    print(f"Task {task_idx+1}: Labels {task_labels}")
    task_data = torch.utils.data.Subset(mnist_train, task_indices.nonzero().squeeze())
    task_loader = DataLoader(task_data, batch_size=32, shuffle=True)

    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(task_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            Lc = loss.item()
            loss.backward()
            # Update the parameters whose corresponding global mask value is 1
            model.fc1.weight.grad.data[global_fc1_mask == 1] = 0
            model.fc2.weight.grad.data[global_fc2_mask == 1] = 0
            model.fc3.weight.grad.data[global_fc3_mask == 1] = 0
            optimizer.step()
            model.fc1.weight.grad.data.zero_()
            model.fc2.weight.grad.data.zero_()
            model.fc3.weight.grad.data.zero_()
            prev_inputs, prev_targets = inputs, targets
            running_loss += Lc
        if epoch == 9:
            fc1_saliency_map, fc2_saliency_map, fc3_saliency_map = compute_saliency_maps(model, inputs.to(device), targets.to(device))
            fc1_binary_mask, fc2_binary_mask, fc3_binary_mask = create_binary_masks(fc1_saliency_map, fc2_saliency_map, fc3_saliency_map, k=10000)
            binary_masks.append((fc1_binary_mask.to(device), fc2_binary_mask.to(device), fc3_binary_mask.to(device)))
            global_fc1_mask = torch.logical_or(global_fc1_mask, fc1_binary_mask.to(device))
            global_fc2_mask = torch.logical_or(global_fc2_mask, fc2_binary_mask.to(device))
            global_fc3_mask = torch.logical_or(global_fc3_mask, fc3_binary_mask.to(device))
            print(torch.sum(global_fc1_mask))
            print(torch.sum(global_fc2_mask))
            print(torch.sum(global_fc3_mask))

        print(f"Epoch {epoch+1}, Task {task_idx+1} train loss: {running_loss / len(task_loader)}")
    # Evaluate the model on the test set for all tasks
    test_task_indices = (mnist_test.targets == task_labels[0]) | (mnist_test.targets == task_labels[1])
    test_data = torch.utils.data.Subset(mnist_test, test_task_indices.nonzero().squeeze())
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        # Apply the matching masks to the model parameters
        model.fc1.weight.data = model.fc1.weight.data * fc1_binary_mask.to(device)
        model.fc2.weight.data = model.fc2.weight.data * fc2_binary_mask.to(device)
        model.fc3.weight.data = model.fc3.weight.data * fc3_binary_mask.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    # Restore the original model parameters
    for name, param in model.named_parameters():
        param.data = original_params[name]
    test_accuracy = (correct / total)*100
    print(f"Task {task_labels} test accuracy: {test_accuracy:.4f}")