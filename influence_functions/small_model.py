import torch as t
import einops
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data import Dataset
import string
import torch
from decoder_only_transformer import DecoderTransformer
import torch as t
from torch.autograd import grad
from utils import save_and_print_results


class CharPredictDataset(Dataset):
    def __init__(self, length=20, seq_length=5):
        self.data = self._generate_data(length)
        self.seq_length = seq_length

    def _generate_data(self, length):
        alphabets = string.ascii_lowercase
        numbers = [str(i % 10) for i in range(length)]

        combined = []
        for i in range(length):
            combined.append(alphabets[i % len(alphabets)])
            combined.append(numbers[i])

        return "".join(combined)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        source_seq = self.data[idx : idx + self.seq_length]
        return torch.tensor([ord(c) for c in source_seq], dtype=torch.long)


class MultiHeadMaskedAttention(t.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = t.nn.Linear(d_model, d_model)
        self.k_proj = t.nn.Linear(d_model, d_model)
        self.v_proj = t.nn.Linear(d_model, d_model)
        self.out_proj = t.nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):
        Q = einops.rearrange(self.q_proj(X), "b s (h d) -> b h s d", h=self.n_heads)
        K = einops.rearrange(self.k_proj(X), "b s (h d) -> b h s d", h=self.n_heads)
        V = einops.rearrange(self.v_proj(X), "b s (h d) -> b h s d", h=self.n_heads)

        # Compute the scaled dot-product attention
        QK = t.einsum("b h i d, b h j d -> b h i j", Q, K)
        QK = QK / t.sqrt(t.tensor(self.d_head))
        if mask is not None:
            QK = QK.masked_fill(mask, -1e9)
        QK = t.nn.functional.softmax(QK, dim=-1)

        # Compute the output
        Y = t.einsum("b h i j, b h j d -> b h i d", QK, V)
        Y = einops.rearrange(Y, "b h s d -> b s (h d)")

        # Apply the output projection
        Y = self.out_proj(Y)

        return Y


class TransformerBlock(t.nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.attn = MultiHeadMaskedAttention(d_model, n_heads)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(d_model, d_mlp),
            t.nn.ReLU(),
            t.nn.Linear(d_mlp, d_model),
        )
        self.layer_norm1 = t.nn.LayerNorm(d_model)
        self.layer_norm2 = t.nn.LayerNorm(d_model)

    def forward(self, X, mask=None):
        attn_output = self.attn(X, mask)
        X = self.layer_norm1(X + attn_output)
        mlp_output = self.mlp(X)
        Y = self.layer_norm2(X + mlp_output)
        return Y


class DecoderTransformer(t.nn.Module):
    def __init__(self, d_model, n_heads, d_mlp, n_layers, vocab_size, max_seq_len=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embed_input = t.nn.Embedding(vocab_size, d_model)
        self.blocks = t.nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)]
        )
        self.out_proj = t.nn.Linear(d_model, vocab_size)

        # Incrementing position embeddings
        self.position_embeddings = t.nn.Embedding(max_seq_len, d_model)

        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def forward(self, X, mask=None):
        X = self.embed_input(X)

        # Add position embeddings
        positions = t.arange(0, X.size(1), device=X.device).unsqueeze(0)
        X = X + self.position_embeddings(positions)

        for block in self.blocks:
            X = block(X, mask)
        Y = self.out_proj(X)
        return Y


def train_loop(model, data_loader, loss_fn, optimizer, num_epochs=2):
    model.train()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)
    print_every = num_epochs // 50
    print_every = 1 if print_every == 0 else print_every
    for epoch in range(num_epochs):
        total_loss = 0
        for sequence in data_loader:
            seq_len = sequence.shape[-1]
            sequence = sequence.to(device)
            mask = (
                t.triu(t.ones(seq_len - 1, seq_len - 1), diagonal=1).bool().to(device)
            )
            model_input = sequence[..., :-1]
            output = model(model_input, mask)
            output = einops.rearrange(output, "b s v -> (b s) v")
            target = sequence[..., 1:]
            target = einops.rearrange(target, "b s -> (b s)")
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def train_char_predict():
    d_model = 120
    n_heads = 4
    d_mlp = 256
    n_layers = 2
    vocab_size = 128
    n_epochs = 500

    small_transformer = DecoderTransformer(
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=5,
    )

    # Assuming CharPredictDataset is defined as before
    dataset = CharPredictDataset(length=20, seq_length=5)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Loss function
    loss_fn = CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(small_transformer.parameters(), lr=0.001)

    train_loop(small_transformer, data_loader, loss_fn, optimizer, num_epochs=n_epochs)

    t.save(small_transformer.state_dict(), "small_transformer.pth")


def get_ekfac_factors(model, train_data):
    """Fit EK-FAC factors to the model on the training set."""

    n_blocks = len(model.blocks)

    P, M = (
        model.blocks[0].mlp[0].weight.shape
    )  # P = Output dimension, M = Input dimension

    # Initialize a list of Kronecker factors for the input covariance.
    kfac_input_covs = [t.zeros(M, M).to(model.device) for _ in range(n_blocks)]

    # Initialize a list of Kronecker factors for the gradient covariance.
    kfac_grad_covs = [t.zeros(P, P).to(model.device) for _ in range(n_blocks)]

    for token_input in train_data:
        seq_len = token_input.shape[-1] - 1
        model_input = token_input.unsqueeze(0).to(model.device)
        embed_input = model_input[..., :-1]
        target = model_input[..., 1:]
        target = einops.rearrange(target, "b s -> (b s)")
        embed_input = model.embed_input(embed_input)

        for idx, block in enumerate(model.blocks):
            mask = t.triu(t.ones(seq_len, seq_len), diagonal=1).bool().to(model.device)
            block_output = block(embed_input, mask=mask)
            # Compute and accumulate the outer product of the inputs to obtain the input covariance.
            kfac_input_covs[idx] += t.einsum(
                "...i,...j->ij",
                embed_input.reshape(-1, embed_input.shape[-1]),
                embed_input.reshape(-1, embed_input.shape[-1]),
            )

            # Compute the gradient of the block loss w.r.t. the block's first MLP layer's weights.
            block_output = einops.rearrange(block_output, "b s v -> (b s) v")

            loss = t.nn.functional.cross_entropy(block_output, target)
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

    n_blocks = len(model.blocks)

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
    seq_len = query.shape[-1] - 1
    target = query[..., 1:].to(model.device)
    target = einops.rearrange(target, "b s -> (b s)")
    embed_query = query[..., :-1].to(model.device)
    embed_query = model.embed_input(embed_query)
    unembed = model.out_proj
    for block in model.blocks:
        mask = t.triu(t.ones(seq_len, seq_len), diagonal=1).bool().to(model.device)
        block_output = block(embed_query, mask=mask)
        block_output = unembed(block_output)
        block_output = einops.rearrange(block_output, "b s v -> (b s) v")
        loss = t.nn.functional.cross_entropy(block_output, target)
        grads = grad(loss, block.mlp[0].weight)[0]
        query_grads.append(grads.reshape(-1))
    return query_grads


def get_train_grads(model, train_seqs):
    n_blocks = len(model.blocks)
    train_grads = [[] for _ in range(n_blocks)]
    unembed = model.out_proj
    for seq in train_seqs:
        seq_len = seq.shape[-1] - 1
        seq = seq.unsqueeze(0).to(model.device)
        target = seq[..., 1:]
        target = einops.rearrange(target, "b s -> (b s)")
        embed_seq = seq[..., :-1]
        embed_seq = model.embed_input(embed_seq)
        for i, block in enumerate(model.blocks):
            mask = t.triu(t.ones(seq_len, seq_len), diagonal=1).bool().to(model.device)
            block_output = block(embed_seq, mask=mask)
            block_output = unembed(block_output)
            block_output = einops.rearrange(block_output, "b s v -> (b s) v")
            loss = t.nn.functional.cross_entropy(block_output, target)
            grads = grad(loss, block.mlp[0].weight)[0]
            train_grads[i].append(grads.reshape(-1))
    return train_grads


def get_influences(ihvp, train_grads):
    """Compute influences using precomputed ihvp"""
    influences = []
    for example_grads in zip(*train_grads):
        influences.append(t.dot(ihvp, t.cat(example_grads)).item())
    return influences


def influence(model_path):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    base_dataset = CharPredictDataset(length=20, seq_length=5)
    train_dataset = []
    for i in range(len(base_dataset)):
        train_dataset.append(base_dataset[i])

    print("Train dataset created")

    d_model = 120
    n_heads = 4
    d_mlp = 256
    n_layers = 2
    vocab_size = 128

    topk = 2

    model = DecoderTransformer(d_model, n_heads, d_mlp, n_layers, vocab_size)

    model.load_state_dict(torch.load(model_path))

    model.to(device)

    model.eval()

    print("Model loaded")

    # Fit EK-FAC
    kfac_factors = get_ekfac_factors(model, train_dataset)

    print("EK-FAC factors computed")

    # Example outputs
    outputs = [
        "e5f6g",
        "i9j0k",
        "m3n4o",
    ]

    token_ids = [
        torch.tensor([ord(c) for c in source_seq], dtype=torch.long).to(device)
        for source_seq in outputs
    ]

    train_grads = get_train_grads(model, train_dataset)

    print("Train gradients computed")

    all_top_seqs = []
    all_influences = []

    for output_ids in token_ids:
        query = output_ids.unsqueeze(0)
        query_grads = get_query_grads(model, query)
        ihvp = get_ekfac_ihvp(model, query_grads, *kfac_factors)
        influences = get_influences(ihvp, train_grads)

        top_idx = torch.topk(torch.tensor(influences), topk)[1]
        top_seqs = [
            "".join([chr(int(i)) for i in train_dataset[idx].tolist()])
            for idx in top_idx.tolist()
        ]

        all_top_seqs.append(top_seqs)
        all_influences.append([influences[index] for index in top_idx.tolist()])

    # Save and print results
    save_and_print_results(outputs, all_influences, all_top_seqs)


def run_model(model_path):
    d_model = 120
    n_heads = 4
    d_mlp = 256
    n_layers = 2
    vocab_size = 128

    model = DecoderTransformer(d_model, n_heads, d_mlp, n_layers, vocab_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)

    while True:
        user_input = input("Enter a string: ")
        if user_input == "exit":
            return
        if len(user_input) > 4:
            user_input = user_input[-4:]
        token_ids = torch.tensor([[ord(c) for c in user_input]], dtype=torch.long).to(
            device
        )
        model_output = model(token_ids)
        last_token = model_output[0, -1, :]
        topk = torch.topk(last_token, 1)
        topk_tokens = [chr(int(i)) for i in topk.indices.tolist()]
        print(topk_tokens[0])


if __name__ == "__main__":
    # train_char_predict()
    # run_model("small_transformer.pth")
    influence("small_transformer.pth")
