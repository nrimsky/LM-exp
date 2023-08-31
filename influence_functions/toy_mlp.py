import torch as t
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import sample
from influence_functions import influence, InfluenceCalculable


# Define the hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 30
hidden_dim = 64
input_dim = 28 * 28  # Downsampled MNIST images are 14x14
output_dim = 10  # 10 classes for digits 0-9
device = t.device("cuda" if t.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        # transforms.Resize((14, 14), antialias=True),  # Downsample to 14x14,
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1],
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten
    ]
)

class MLPBlock(InfluenceCalculable, t.nn.Module):

    def __init__(self, input_dim, output_dim, use_relu=True):
        super().__init__()
        self.linear = t.nn.Linear(input_dim, output_dim)
        self.relu = t.nn.ReLU()
        self.input = None
        self.use_relu = use_relu

    def forward(self, x):
        self.input = x
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        return x
    
    def get_weights(self):
        return self.linear.weight
    
    def get_bias(self):
        return self.linear.bias
    
    def get_input(self):
        return self.input



class MLP(t.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = MLPBlock(input_dim, hidden_dim)
        self.fc2 = MLPBlock(hidden_dim, hidden_dim)
        self.fc3 = MLPBlock(hidden_dim, output_dim, use_relu=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_model():
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the MLP model
    model = MLP(input_dim, output_dim, hidden_dim)
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = t.nn.CrossEntropyLoss()

    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # Reset gradients
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    # Testing the model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with t.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = t.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(
        f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%"
    )

    # Save the model checkpoint
    t.save(model.state_dict(), "model.ckpt")

    return model, train_dataset, test_dataset


def run_influence(model_path):
    model = MLP(input_dim, output_dim, hidden_dim)
    model.load_state_dict(t.load(model_path))
    model = model.to(device)

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_subset = t.utils.data.Subset(train_dataset, sample(range(len(train_dataset)), 2000))

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
    test_subset = t.utils.data.Subset(test_dataset, sample(range(len(test_dataset)), 10))

    # mlp_blocks = [model.fc1, model.fc2, model.fc3]
    mlp_blocks = [model.fc2]

    loss_fn = t.nn.CrossEntropyLoss()

    influence(model, mlp_blocks, loss_fn, test_subset, train_subset, device)



if __name__ == "__main__":
    # train_model()
    run_influence("model.ckpt")
