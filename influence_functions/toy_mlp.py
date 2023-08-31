import torch as t
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import sample


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


class MLP(t.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = t.nn.Linear(input_dim, hidden_dim)
        self.fc2 = t.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = t.nn.Linear(hidden_dim, output_dim)
        self.relu = t.nn.ReLU()

        self.s1 = None
        self.s2 = None
        self.s3 = None

        self.a1 = None
        self.a2 = None
        self.a3 = None

    def forward(self, x):
        s1 = self.fc1(x)
        self.s1 = s1
        a1 = self.relu(s1)
        self.a1 = a1

        s2 = self.fc2(a1)
        self.s2 = s2
        a2 = self.relu(s2)
        self.a2 = a2

        s3 = self.fc3(a2)
        self.s3 = s3
        a3 = self.relu(s3)
        self.a3 = a3

        return a3


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


def get_ekfac_factors_and_train_grads(model, dataset):

    train_grads = [[] for _ in range(3)]

    A_lminus1 = [
        t.zeros((M + 1, M + 1)).to(device)
        for M in [
            model.fc1.weight.shape[1],
            model.fc2.weight.shape[1],
            model.fc3.weight.shape[1],
        ]
    ]
    S_l = [
        t.zeros((P, P)).to(device)
        for P in [
            model.fc1.weight.shape[0],
            model.fc2.weight.shape[0],
            model.fc3.weight.shape[0],
        ]
    ]

    tot = 0

    for i, (data, target) in enumerate(dataset):
        model.zero_grad()
        model.train()
        data = data.to(device)
        target = t.tensor(target).to(device)
        output = model(data)
        loss = t.nn.functional.cross_entropy(output.unsqueeze(0), target.unsqueeze(0))

        homog_data = t.cat([data, t.tensor([1]).to(device)], dim=-1)
        a1_cov = t.einsum("i,j->ij", homog_data, homog_data)

        homog_a1 = t.cat([model.a1, t.tensor([1]).to(device)], dim=-1)
        a2_cov = t.einsum("i,j->ij", homog_a1, homog_a1)

        homog_a2 = t.cat([model.a2, t.tensor([1]).to(device)], dim=-1)
        a3_cov = t.einsum("i,j->ij", homog_a2, homog_a2)

        loss.backward()

        ds1 = t.einsum("ik,jk->ij", model.fc1.weight.grad, model.fc1.weight.grad)
        ds2 = t.einsum("ik,jk->ij", model.fc2.weight.grad, model.fc2.weight.grad)
        ds3 = t.einsum("ik,jk->ij", model.fc3.weight.grad, model.fc3.weight.grad)

        train_grads[0].append(t.cat([model.fc1.weight.grad.view(-1), model.fc1.bias.grad.view(-1)]))
        train_grads[1].append(t.cat([model.fc2.weight.grad.view(-1), model.fc2.bias.grad.view(-1)]))
        train_grads[2].append(t.cat([model.fc3.weight.grad.view(-1), model.fc3.bias.grad.view(-1)]))

        A_lminus1[0] += a1_cov
        A_lminus1[1] += a2_cov
        A_lminus1[2] += a3_cov

        S_l[0] += ds1
        S_l[1] += ds2
        S_l[2] += ds3

        tot += 1

    A_lminus1 = [A / tot for A in A_lminus1]
    S_l = [S / tot for S in S_l]

    return A_lminus1, S_l, train_grads


def get_ekfac_ihvp(query_grads, kfac_input_covs, kfac_grad_covs, damping=0.01):
    """Compute EK-FAC inverse Hessian-vector products."""
    ihvp = []
    for i in range(len(query_grads)):
        q = query_grads[i]
        P = kfac_grad_covs[i].shape[0]
        M = kfac_input_covs[i].shape[0]
        q = q.reshape((P, M))
        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, lambda_a, q_a_t = t.svd(kfac_input_covs[i])
        q_s, lambda_s, q_s_t = t.svd(kfac_grad_covs[i])
        # Compute the diagonal matrix with damped eigenvalues
        ekfacDiag = t.outer(
            lambda_a, lambda_s
        ).flatten()  # The Kronecker product's eigenvalues
        ekfacDiag_damped_inv = 1.0 / (ekfacDiag + damping)
        # Reshape the inverted diagonal to match the shape of V (reshaped query gradients)
        reshaped_diag_inv = ekfacDiag_damped_inv.reshape(P, M)
        intermediate_result = q_s @ (q @ q_a_t)
        result = intermediate_result / reshaped_diag_inv
        ihvp_component = q_s_t @ (result @ q_a)
        ihvp.append(ihvp_component.reshape(-1))
    # Concatenating the results across blocks to get the final ihvp
    return t.cat(ihvp)


def get_query_grads(model, query, label):
    model.zero_grad()
    model.train()
    query = query.to(device)
    label = t.tensor(label).to(device)
    output = model(query)
    loss = t.nn.functional.cross_entropy(output.unsqueeze(0), label.unsqueeze(0))
    loss.backward()
    grads = []
    grads.append(t.cat([model.fc1.weight.grad.view(-1), model.fc1.bias.grad.view(-1)]))
    grads.append(t.cat([model.fc2.weight.grad.view(-1), model.fc2.bias.grad.view(-1)]))
    grads.append(t.cat([model.fc3.weight.grad.view(-1), model.fc3.bias.grad.view(-1)]))
    return grads


def get_influences(ihvp, train_grads):
    """Compute influences using precomputed ihvp"""
    influences = []
    for example_grads in zip(*train_grads):
        influences.append(t.dot(ihvp, t.cat(example_grads)).item())
    return influences


def influence(model_path):
    model = MLP(input_dim, output_dim, hidden_dim)
    model.load_state_dict(t.load(model_path))
    model = model.to(device)

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_subset = t.utils.data.Subset(train_dataset, sample(range(len(train_dataset)), 8000))

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
    test_subset = t.utils.data.Subset(test_dataset, sample(range(len(test_dataset)), 10))

    A_lminus1, S_l, train_grads = get_ekfac_factors_and_train_grads(model, train_subset)

    all_top_images = []
    all_influences = []

    for query, label in test_subset:
        query_grads = get_query_grads(model, query, label)
        ihvp = get_ekfac_ihvp(query_grads, A_lminus1, S_l)
        influences = get_influences(ihvp, train_grads)
        top_influences, top_images = t.topk(t.tensor(influences), 10)
        all_top_images.append(top_images)
        all_influences.append(top_influences)

    # Print labels of top images and their respective influences

    for i, (top_images, top_influences) in enumerate(zip(all_top_images, all_influences)):
        print(f"Query label: {test_subset[i][1]}")

        for image_idx, influence in zip(top_images, top_influences):
            print(f"Image label {train_subset[image_idx][1]}: {influence:.4f}")



if __name__ == "__main__":
    # train_model()
    influence("model.ckpt")
