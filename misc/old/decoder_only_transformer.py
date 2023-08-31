import torch as t
import einops
import math
from torchtext.datasets import WikiText103
import torch.optim as optim
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from influence_functions.old.utils import slice_chain_dataset

d_model = 512
n_heads = 8
d_mlp = 2048
n_layers = 8


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.data = [tokenizer.encode(text) for text in data]
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_text = self.data[idx]
        x = t.tensor(tokenized_text[:-1])
        y = t.tensor(tokenized_text[1:])
        x = F.pad(x, (0, self.block_size - len(x)), "constant", 0).long()
        y = F.pad(y, (0, self.block_size - len(y)), "constant", 0).long()
        return x, y


class BPETokenizer:
    def __init__(self, max_vocab_size=50000):
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(special_tokens=["<UNK>"], vocab_size=max_vocab_size)

    def fit_on_texts(self, texts):
        self.tokenizer.train_from_iterator(texts, self.trainer)

    def texts_to_sequences(self, texts):
        return [self.tokenizer.encode(text).ids for text in texts]

    def sequences_to_texts(self, sequences):
        return [self.tokenizer.decode(seq) for seq in sequences]

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path):
        tokenizer_instance = cls()
        tokenizer_instance.tokenizer = Tokenizer.from_file(path)
        return tokenizer_instance

    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)


def get_positional_encoding(d_model, length):
    """
    Get the positional encoding for a sequence of a given length and model size.
    """
    pos = t.arange(length).unsqueeze(1)
    div_term = t.exp(t.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = t.zeros(length, d_model)
    pe[:, 0::2] = t.sin(pos * div_term)
    pe[:, 1::2] = t.cos(pos * div_term)
    return pe


class MultiQueryAttention(t.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_linear = t.nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_linear = t.nn.Linear(d_model, self.head_dim, bias=False)
        self.v_linear = t.nn.Linear(d_model, self.head_dim, bias=False)
        self.o_linear = t.nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, X):
        Q = self.q_linear(X)
        # Reshaping Q to introduce the head dimension
        Q = einops.rearrange(Q, "b n (h d) -> b h n d", h=self.n_heads)
        # Shared Key and Value (without head dimension)
        K = self.k_linear(X)
        V = self.v_linear(X)
        # Logits calculation
        logits = einops.einsum(Q, K, "b h n d, b n d -> b h n d")
        logits /= math.sqrt(self.head_dim)
        mask = t.triu(t.ones(logits.shape[-2:]), diagonal=1).bool().to(X.device)
        logits.masked_fill_(mask, float("-inf"))
        # Softmax over last dimension
        weights = t.nn.functional.softmax(logits, dim=-1)
        # Output calculation
        O = einops.einsum(weights, V, "b h n d, b n d -> b h n d")
        # Reshaping O to remove the head dimension and then transforming through o_linear
        O = einops.rearrange(O, "b h n d -> b n (h d)")
        Y = self.o_linear(O)
        return Y


class TransformerBlock(t.nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.multi_query_attn = MultiQueryAttention(d_model, n_heads)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(d_model, d_mlp),
            t.nn.ReLU(),
            t.nn.Linear(d_mlp, d_model),
        )
        self.layer_norm1 = t.nn.LayerNorm(d_model)
        self.layer_norm2 = t.nn.LayerNorm(d_model)

    def forward(self, X):
        # Multi-query attention
        Y = self.multi_query_attn(X)
        # Residual connection and layer norm
        Y = self.layer_norm1(X + Y)
        # MLP
        Z = self.mlp(Y)
        # Residual connection and layer norm
        Z = self.layer_norm2(Y + Z)
        return Z


class DecoderTransformer(t.nn.Module):
    def __init__(self, d_model, n_heads, d_mlp, n_layers, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.blocks = t.nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)]
        )
        self.embed = t.nn.Embedding(vocab_size, d_model)
        self.out_proj = t.nn.Linear(d_model, vocab_size)

    def embed_input(self, X):
        tok_emb = self.embed(X) * math.sqrt(self.d_model)
        pos_emb = get_positional_encoding(self.d_model, X.shape[1]).to(X.device)
        total_emb = tok_emb + pos_emb
        return total_emb

    def forward(self, X):
        emb = self.embed_input(X)
        Y = self.blocks(emb)
        Y = self.out_proj(Y)
        return Y

    def generate(self, X, max_len=20):
        self.eval()
        with t.no_grad():
            for _ in range(max_len):
                Y = self(X)
                Y = t.nn.functional.softmax(Y[:, -1, :], dim=-1)
                next_token = t.multinomial(Y, num_samples=1)
                X = t.cat([X, next_token], dim=-1)
        return X


def get_train_loader(make_new_tokenizer=False, limit=None, batch_size=64):
    train_data, valid_data, test_data = WikiText103()
    train_data = train_data + valid_data + test_data

    if limit is not None:
        train_data = slice_chain_dataset(train_data, limit)

    if os.path.exists("bpe_tokenizer.json") and not make_new_tokenizer:
        tokenizer = BPETokenizer.load("bpe_tokenizer.json")
    else:
        tokenizer = BPETokenizer()
        tokenizer.fit_on_texts(train_data)
        tokenizer_path = "bpe_tokenizer.json"
        tokenizer.save(tokenizer_path)

    # Convert tokenized data to sequences
    block_size = 120
    train_dataset = TextDataset(train_data, tokenizer, block_size)

    # Create iterators
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Combine everything in train_loader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return train_loader, train_dataset, tokenizer


def train():
    train_loader, _, tokenizer = get_train_loader(make_new_tokenizer=True)

    # Initialize the model
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    model = DecoderTransformer(
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
        vocab_size=tokenizer.get_vocab_size(),
    )
    if os.path.exists("final_model.pth"):
        model.load_state_dict(t.load("final_model.pth"))

    print(count_parameters(model), "params")

    model = model.to(device)
    num_epochs = 1

    # Loss and Optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    warmup_steps = 10_000
    total_steps = len(train_loader) * num_epochs

    scheduler = t.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # Training Loop
    save_every = len(train_loader) // 20  # Save model every 'save_every' batches

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        for idx, (data, target) in pbar:
            data, target = data.long().to(device), target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(
                output.view(-1, tokenizer.get_vocab_size()), target.view(-1)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=loss.item())

            # Save checkpoints and display the average loss so far
            if idx % save_every == 0:
                t.save(
                    model.state_dict(),
                    f"checkpoints/checkpoint_epoch{epoch}_batch{idx}.pth",
                )

    # save the model
    t.save(model.state_dict(), "final_model.pth")
    return model


def loop():
    """
    Interactive loop to generate text from the model.
    """
    tokenizer = BPETokenizer.load("bpe_tokenizer.json")

    # Initialize the model
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    model = DecoderTransformer(
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
        vocab_size=tokenizer.get_vocab_size(),
    )
    model = model.to(device)
    model.load_state_dict(t.load("final_model.pth"))

    while True:
        inp = input("Enter a prompt or Q to quit: ")
        if inp.strip().lower() == "q":
            break
        inp = tokenizer.texts_to_sequences([inp])[0]
        inp = t.tensor(inp).unsqueeze(0).to(device)
        out = model.generate(inp)
        out = tokenizer.sequences_to_texts(out.tolist())[0]
        print(out)
