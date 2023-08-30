import torch
import matplotlib.pyplot as plt
from decoder_only_transformer import DecoderTransformer, get_train_loader
import torch as t
from torch.autograd import grad


def get_ekfac_factors(model, train_data):
    """Fit EK-FAC factors to the model on the training set."""

    P, M = (
        model.blocks[0].mlp[0].weight.shape
    )  # P = Output dimension, M = Input dimension
    n_mlps = len(model.blocks)

    # Initialize a list of Kronecker factors for the input covariance.
    kfac_input_covs = [t.zeros(M, M) for _ in range(n_mlps)]

    # Initialize a list of Kronecker factors for the gradient covariance.
    kfac_grad_covs = [t.zeros(P, P) for _ in range(n_mlps)]

    for token_input, _ in train_data:
        block_input = model.embed_input(token_input.unsqueeze(0))
        for idx, block in enumerate(model.blocks):
            block_output = block(block_input)
            # Compute and accumulate the outer product of the inputs to obtain the input covariance.
            kfac_input_covs[idx] += t.einsum("...i,...j->ij", block_input, block_input)

            # Compute the gradient of the block loss w.r.t. the block's first MLP layer's weights.
            block_output, block_input = (
                block_output[..., :-1, :],
                block_input[..., 1:, :],
            )
            loss = t.nn.functional.cross_entropy(block_output, block_input)
            block_grad_weights = grad(loss, block.mlp[0].weight)[0]

            # Use this gradient for computing the kfac_grad_covs
            kfac_grad_covs[idx] += t.einsum(
                "ik,jk->ij", block_grad_weights, block_grad_weights
            )

    # Normalize the accumulated Kronecker factors by the number of batches in the train loader.
    for idx in range(len(model.blocks)):
        kfac_input_covs[idx] /= len(train_data)
        kfac_grad_covs[idx] /= len(train_data)

    # Return the computed Kronecker factors for input and gradient covariances.
    return kfac_input_covs, kfac_grad_covs


def get_ekfac_ihvp(model, query_grads, kfac_input_covs, kfac_grad_covs, damping=0.01):
    """Compute EK-FAC inverse Hessian-vector products."""

    ihvp = []
    P, M = (
        model.blocks[0].mlp[0].weight.shape
    )  # P = Output dimension, M = Input dimension
    n_mlps = len(model.blocks)

    for i in range(n_mlps):
        q = query_grads[i].reshape((P, M))

        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, lambda_a, q_a_t = t.svd(kfac_input_covs[i])
        q_s, lambda_s, q_s_t = t.svd(kfac_grad_covs[i])

        assert lambda_a.shape == (M,)
        assert lambda_s.shape == (P,)

        # Compute the diagonal matrix with damped eigenvalues
        ekfacDiag = t.outer(
            lambda_a, lambda_s
        ).flatten()  # The Kronecker product's eigenvalues
        ekfacDiag_damped_inv = 1.0 / (ekfacDiag + damping)

        assert ekfacDiag_damped_inv.shape == (M * P,)

        # Reshape the inverted diagonal to match the shape of V (reshaped query gradients)
        reshaped_diag_inv = ekfacDiag_damped_inv.reshape(P, M)

        intermediate_result = q_s @ (q @ q_a_t)
        result = intermediate_result / reshaped_diag_inv
        ihvp_component = q_s_t @ (result @ q_a)

        ihvp.append(ihvp_component.reshape(-1))

    # Concatenating the results across blocks to get the final ihvp
    return t.cat(ihvp)


def get_query_grads(model, query):
    """Get query grad"""

    query_grads = []
    query = model.embed_input(query)

    for block in model.blocks:
        block.zero_grad()
        block_output = block(query)

        block_output, query = block_output[..., :-1, :], query[..., 1:, :]
        loss = t.nn.functional.cross_entropy(block_output, query)
        grads = grad(loss, block.mlp[0].weight)[0]
        query_grads.append(grads.reshape(-1))

    return query_grads


def get_train_grads(model, train_seqs):
    """Get gradients for a batch of training sequences."""

    train_grads = [[] for _ in model.blocks]

    for seq, _ in train_seqs:
        seq = model.embed_input(seq.unsqueeze(0))
        for i, block in enumerate(model.blocks):
            block.zero_grad()
            block_output = block(seq)

            block_output, seq = block_output[..., :-1, :], seq[..., 1:, :]
            loss = t.nn.functional.cross_entropy(block_output, seq)
            grads = grad(loss, block.mlp[0].weight)[0]
            train_grads[i].append(grads.reshape(-1))

    return train_grads


def get_influences(model, query, train_seqs, kfac_factors):
    """Compute influence scores n the query."""

    query_grads = get_query_grads(model, query)
    ihvp = get_ekfac_ihvp(model, query_grads, *kfac_factors)

    influences = []
    train_grads = get_train_grads(model, train_seqs)

    for example_grads in zip(*train_grads):
        influences.append(t.dot(ihvp, t.cat(example_grads)).item())

    return influences


# Model hyperparameters
d_model = 512
n_heads = 8
d_mlp = 2048
n_layers = 8

_, train_dataset, tokenizer = get_train_loader(make_new_tokenizer=False, limit=5)

# Load pretrained model
model = DecoderTransformer(
    d_model, n_heads, d_mlp, n_layers, tokenizer.get_vocab_size()
)
model.load_state_dict(torch.load("final_model.pth"))

# Fit EK-FAC
kfac_factors = get_ekfac_factors(model, train_dataset)

print("EK-FAC factors")

# Example outputs
outputs = [
    "The first president of the United States was George Washington.",
    "My favorite food is pizza.",
    "The capital of France is Paris.",
]

# Tokenize outputs
token_ids = [tokenizer.encode(output) for output in outputs]
# Plotting
fig, axs = plt.subplots(
    len(outputs), 1, figsize=(10, 15)
)  # Adjust the figure size as needed

topk = 3
for i, output_ids in enumerate(token_ids):
    # Get query grads
    query = torch.tensor([output_ids])

    # Get influences
    influences = get_influences(model, query, train_dataset, kfac_factors)

    # Top influencing sequences
    top_idx = torch.topk(torch.tensor(influences), topk)[1]
    top_seqs = [
        tokenizer.decode(train_dataset[idx][0].tolist()) for idx in top_idx.tolist()
    ]

    # Get the influences for the top sequences only
    heatmaps = [influences[index] for index in top_idx.tolist()]

    # Display the heatmap
    im = axs[i].imshow(
        [heatmaps],
        aspect="auto",
        cmap="coolwarm",
        vmin=min(heatmaps),
        vmax=max(heatmaps),
    )
    axs[i].set_title(outputs[i])

    # Display the token sequences alongside the heatmap
    axs[i].set_yticks(range(len(top_seqs)))
    axs[i].set_yticklabels(top_seqs)

fig.colorbar(im, ax=axs[:])
plt.savefig("heatmap_tokens.png")
