import torch as t
import einops
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data import Dataset
import string
import torch
from influence_functions import influence, InfluenceCalculable

d_model = 120
n_heads = 4
d_mlp = 256
n_layers = 2
vocab_size = 128
dataset_length = 20
sequence_length = 5


def autoregressive_loss(output, target):
    output = einops.rearrange(output, "b s v -> (b s) v")
    target = einops.rearrange(target, "b s -> (b s)")
    loss = t.nn.functional.cross_entropy(output, target)
    return loss


class CharPredictDataset(Dataset):
    def __init__(self, length, seq_length):
        self.data = self._generate_data(length)
        self.seq_length = seq_length

    def _generate_data(self, length):
        alphabets = string.ascii_lowercase
        numbers = [str(i % 10) for i in range(length)]
        return "".join([alphabets[i] + numbers[i] for i in range(length)])

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        source_seq = self.data[idx : idx + self.seq_length]
        return torch.tensor(
            [ord(c) for c in source_seq[:-1]], dtype=torch.long
        ), torch.tensor([ord(c) for c in source_seq[1:]], dtype=torch.long)


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


class MLPBlock(InfluenceCalculable, t.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = t.nn.Linear(input_dim, hidden_dim)
        self.relu = t.nn.ReLU()
        self.linear2 = t.nn.Linear(hidden_dim, output_dim)
        self.input = None

    def forward(self, x):
        self.input = x
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def get_weights(self):
        return self.linear.weight

    def get_bias(self):
        return self.linear.bias

    def get_input(self):
        return self.input


class TransformerBlock(t.nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.attn = MultiHeadMaskedAttention(d_model, n_heads)
        self.mlp = MLPBlock(d_model, d_mlp, d_model)
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

    def forward(self, X):
        seq_len = X.size(-1)

        mask = t.triu(t.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
        X = self.embed_input(X)
        # Add position embeddings
        positions = t.arange(0, seq_len, device=X.device).unsqueeze(0)
        X = X + self.position_embeddings(positions)

        for block in self.blocks:
            X = block(X, mask)
        Y = self.out_proj(X)
        return Y


def train_loop(model, data_loader, optimizer, num_epochs=2):
    model.train()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)
    print_every = num_epochs // 50
    print_every = 1 if print_every == 0 else print_every
    for epoch in range(num_epochs):
        total_loss = 0
        for model_input, target in data_loader:
            model_input, target = model_input.to(device), target.to(device)
            output = model(model_input)
            loss = autoregressive_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def train_char_predict(n_epochs=500):
    small_transformer = DecoderTransformer(
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=5,
    )

    # Assuming CharPredictDataset is defined as before
    dataset = CharPredictDataset(length=dataset_length, seq_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Optimizer
    optimizer = Adam(small_transformer.parameters(), lr=0.001)

    train_loop(small_transformer, data_loader, optimizer, num_epochs=n_epochs)

    t.save(small_transformer.state_dict(), "small_transformer.pth")


def calc_influence(model_path):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    base_dataset = CharPredictDataset(length=dataset_length, seq_length=sequence_length)
    train_dataset = []
    for input, target in base_dataset:
        train_dataset.append((input, target[-1].item()))

    print("Train dataset created")

    model = DecoderTransformer(d_model, n_heads, d_mlp, n_layers, vocab_size)

    model.load_state_dict(torch.load(model_path))

    model.to(device)

    model.train()

    def encode(string):
        return torch.tensor([ord(c) for c in string], dtype=torch.long).to(device)

    # Example outputs
    queries = [
        (encode("e5f6"), encode("g")),
        (encode("a1b2"), encode("c")),
        (encode("z9y8"), encode("x")),
        (encode("m3n4"), encode("o")),
        (encode("j7k8"), encode("l")),
        (encode("q5r6"), encode("s")),
    ]

    ce = CrossEntropyLoss()

    def loss_fn(output, target):
        return ce(output[:, -1, :], target)

    all_top_training_samples, all_top_influences = influence(
        model,
        [b.mlp for b in model.blocks],
        loss_fn,
        queries,
        train_dataset,
        device,
    )

    def decode(token_ids):
        try:
            return "".join([chr(i) for i in token_ids])
        except:
            return chr(token_ids)

    for i, (top_samples, top_influences) in enumerate(
        zip(all_top_training_samples, all_top_influences)
    ):
        print(f"Query: Input {decode(queries[i][0])} Target: {decode(queries[i][1])}")
        print("Top 10 training samples and their influences:")
        for s, i in zip(top_samples, top_influences):
            s = s.item()
            print(
                f"Sample: {decode(train_dataset[s][0])} {decode(train_dataset[s][1])} Influence: {i}"
            )


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
    run_model("small_transformer.pth")
    calc_influence("small_transformer.pth")
