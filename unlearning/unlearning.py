import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

def unlearning(net, retain, forget, validation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 8
    names = [o[0] for o in list(net.named_parameters())][-15:]
    params_to_update = []
    for name, param in net.named_parameters():
        if name in names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    print("Params to learn:", len(params_to_update))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=0.001)
    net.train()

    retain_iter = iter(retain)
    forget_iter = iter(forget)

    for _ in range(epochs):
        for idx in tqdm(range(max(len(retain), len(forget)))):
            if idx % 2 == 0:
                try:
                    inputs, targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain)
                    inputs, targets = next(retain_iter)
                use_retain = True
            else:
                try:
                    inputs, targets = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget)
                    inputs, targets = next(forget_iter)
                use_retain = False

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if use_retain:
                loss = criterion(outputs, targets)
            else:
                # One-hot encoding targets and add noise
                num_classes = outputs.size()[1]
                targets_noisy = (F.one_hot(targets, num_classes).float() + 5 * torch.randn(targets.size(0), num_classes).to(device))/6
                # Normalize to make it a distribution
                targets_noisy = F.softmax(targets_noisy, dim=1)
                # Use softmax for probabilities
                outputs_soft = F.softmax(outputs, dim=1)
                # Add a KL regularization term
                loss = 2 * F.kl_div(outputs_soft.log(), targets_noisy, reduction='batchmean')
            loss.backward()
            optimizer.step()
    net.eval()
    return net