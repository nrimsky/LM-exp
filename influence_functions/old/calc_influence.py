import torch
from decoder_only_transformer import DecoderTransformer, get_train_loader
import torch as t
from torch.autograd import grad
from influence_functions.old.utils import save_and_print_results

# Model Hyperparameters
d_model = 512
n_heads = 8
d_mlp = 2048
n_layers = 8

# Influence Function Hyperparameters
n_blocks = 3  # Number of final blocks to compute influences for
train_samples = 200  # Number of training samples to use
topk = 3  # Number of top influential sequences to return

def get_ekfac_factors(model, train_data):
    """Fit EK-FAC factors to the model on the training set."""

    P, M = (
        model.blocks[0].mlp[0].weight.shape
    )  # P = Output dimension, M = Input dimension

    # Initialize a list of Kronecker factors for the input covariance.
    kfac_input_covs = [t.zeros(M, M) for _ in range(n_blocks)]

    # Initialize a list of Kronecker factors for the gradient covariance.
    kfac_grad_covs = [t.zeros(P, P) for _ in range(n_blocks)]

    for token_input, _ in train_data:
        block_input = model.embed_input(token_input.unsqueeze(0))
        for idx, block in enumerate(model.blocks[-n_blocks:]):
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
    for idx in range(n_blocks):
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

    for i in range(n_blocks):
        q = query_grads[i].reshape((P, M))

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


def get_query_grads(model, query):
    query_grads = []
    query = model.embed_input(query)
    for block in model.blocks[-n_blocks:]:
        block_output = block(query)
        block_output, query = block_output[..., :-1, :], query[..., 1:, :]
        loss = t.nn.functional.cross_entropy(block_output, query)
        grads = grad(loss, block.mlp[0].weight)[0]
        query_grads.append(grads.reshape(-1))
    return query_grads


def get_train_grads(model, train_seqs):
    train_grads = [[] for _ in range(n_blocks)]
    for seq, _ in train_seqs:
        seq = model.embed_input(seq.unsqueeze(0))
        for i, block in enumerate(model.blocks[-n_blocks:]):
            block_output = block(seq)
            block_output, seq = block_output[..., :-1, :], seq[..., 1:, :]
            loss = t.nn.functional.cross_entropy(block_output, seq)
            grads = grad(loss, block.mlp[0].weight)[0]
            train_grads[i].append(grads.reshape(-1))
    return train_grads


def get_influences(ihvp, train_grads):
    """Compute influences using precomputed ihvp"""
    influences = []
    for example_grads in zip(*train_grads):
        influences.append(t.dot(ihvp, t.cat(example_grads)).item())
    return influences

_, train_dataset, tokenizer = get_train_loader(make_new_tokenizer=False, limit=train_samples)

# Load pretrained model
model = DecoderTransformer(
    d_model, n_heads, d_mlp, n_layers, tokenizer.get_vocab_size()
)
model.load_state_dict(torch.load("final_model.pth"))

# Fit EK-FAC
kfac_factors = get_ekfac_factors(model, train_dataset)

print("EK-FAC factors computed")

# Example outputs
outputs = [
    "Partly due to these events, I like oranges.",
    "I really like manga and anime.",
    "Fighting in the army is not fun.",
]

# Tokenize outputs
token_ids = [tokenizer.encode(output) for output in outputs]

train_grads = get_train_grads(model, train_dataset)

print("Train gradients computed")

all_top_seqs = []
all_influences = []

for i, output_ids in enumerate(token_ids):
    query = torch.tensor([output_ids])
    query_grads = get_query_grads(model, query)
    ihvp = get_ekfac_ihvp(model, query_grads, *kfac_factors)
    influences = get_influences(ihvp, train_grads)
    
    top_idx = torch.topk(torch.tensor(influences), topk)[1]
    top_seqs = [tokenizer.decode(train_dataset[idx][0].tolist()) for idx in top_idx.tolist()]
    
    all_top_seqs.append(top_seqs)
    all_influences.append([influences[index] for index in top_idx.tolist()])

# Save and print results
save_and_print_results(outputs, all_influences, all_top_seqs)