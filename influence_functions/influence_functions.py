import torch as t
from abc import ABC, abstractmethod
from typing import List


class InfluenceCalculable(ABC):
    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def get_bias(self):
        pass

    @abstractmethod
    def get_input(self):
        pass


def get_ekfac_factors_and_train_grads(
    model, dataset, loss_fn, mlp_blocks: List[InfluenceCalculable], device
):
    kfac_input_covs = [
        t.zeros((b.get_weights().shape[1] + 1, b.get_weights().shape[1] + 1)).to(device)
        for b in mlp_blocks
    ]
    kfac_grad_covs = [
        t.zeros((b.get_weights().shape[0], b.get_weights().shape[0])).to(device)
        for b in mlp_blocks
    ]
    train_grads = [[] for _ in range(len(mlp_blocks))]
    tot = 0
    for i, (data, target) in enumerate(dataset):
        model.zero_grad()
        model.train()
        data = data.to(device)
        target = t.tensor(target).to(device)
        if len(data.shape) == 1:
            # Add batch dimension
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
        output = model(data)
        loss = loss_fn(output, target)
        for i, block in enumerate(mlp_blocks):
            homog_input = t.cat(
                [block.get_input(), t.ones((block.get_input().shape[0], 1)).to(device)],
                dim=-1,
            )
            input_cov = t.einsum("...i,...j->ij", homog_input, homog_input)
            kfac_input_covs[i] += input_cov
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            weights_grad = block.get_weights().grad
            bias_grad = block.get_bias().grad
            grad_cov = t.einsum("...ik,...jk->ij", weights_grad, weights_grad)
            kfac_grad_covs[i] += grad_cov
            train_grads[i].append(t.cat([weights_grad.view(-1), bias_grad.view(-1)]))
        tot += 1
    kfac_input_covs = [A / tot for A in kfac_input_covs]
    kfac_grad_covs = [S / tot for S in kfac_grad_covs]
    return kfac_input_covs, kfac_grad_covs, train_grads


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


def get_query_grads(
    model, query, target, mlp_blocks: List[InfluenceCalculable], loss_fn, device
):
    model.zero_grad()
    model.train()
    query = query.to(device)
    target = t.tensor(target).to(device)
    output = model(query)
    loss = loss_fn(output.unsqueeze(0), target.unsqueeze(0))
    loss.backward()
    grads = []
    for block in mlp_blocks:
        weights_grad = block.get_weights().grad
        bias_grad = block.get_bias().grad
        grads.append(t.cat([weights_grad.view(-1), bias_grad.view(-1)]))
    return grads


def get_influences(ihvp, train_grads):
    """Compute influences using precomputed ihvp"""
    influences = []
    for example_grads in zip(*train_grads):
        influences.append(t.dot(ihvp, t.cat(example_grads)).item())
    return influences


def influence(
    model,
    mlp_blocks: List[InfluenceCalculable],
    loss_fn,
    test_dataset,
    train_dataset,
    device,
):
    loss_fn = t.nn.CrossEntropyLoss()
    kfac_input_covs, kfac_grad_covs, train_grads = get_ekfac_factors_and_train_grads(
        model, train_dataset, loss_fn, mlp_blocks, device
    )

    all_top_training_samples = []
    all_top_influences = []

    for query, target in test_dataset:
        query_grads = get_query_grads(model, query, target, mlp_blocks, loss_fn, device)
        ihvp = get_ekfac_ihvp(query_grads, kfac_input_covs, kfac_grad_covs)
        top_influences = get_influences(ihvp, train_grads)
        top_influences, top_samples = t.topk(t.tensor(top_influences), 10)
        all_top_training_samples.append(top_samples)
        all_top_influences.append(top_influences)

    for i, (top_samples, top_influences) in enumerate(
        zip(all_top_training_samples, all_top_influences)
    ):
        print(f"Query target: {test_dataset[i][1]}")

        for sample_idx, influence in zip(top_samples, top_influences):
            print(f"Sample target {train_dataset[sample_idx][1]}: {influence:.4f}")
