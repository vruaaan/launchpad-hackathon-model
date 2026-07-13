from autoregression.data.dataprep import N_TOKENS, PAD_ID, EOS_ID, SOS_ID, TOKEN_TO_ID, ID_TO_TOKEN, collate
from autoregression.data.dataprep import SQLSeqDataset
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LSTMCore(nn.Module):
    """Embedding -> multi-layer LSTM -> Linear projection to vocab logits."""
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)
 
    def forward(self, input_ids, hidden=None):
        x = self.embedding(input_ids)            # (B, T, emb_dim)
        out, hidden = self.lstm(x, hidden)        # (B, T, hidden_dim)
        out = self.dropout(out)
        logits = self.head(out)                   # (B, T, vocab)
        return logits, hidden
    


class RNNAgent:
    """
    Owns the LSTM model plus everything needed to train, sample from,
    and evaluate it on the SQL-token sequences.
    Usage:
        agent = Agent(vocab_size=N_TOKENS, pad_id=PAD_ID,
                       sos_id=SOS_ID, eos_id=EOS_ID)
        agent.fit(training_tensors, validation_tensors, collate_fn=collate, epochs=20)
        ppl = agent.evaluate(validation_tensors, collate_fn=collate)
        tokens = agent.sample(id_to_token=ID_TO_TOKEN, max_len=40)
    """
    def __init__(
        self,
        vocab_size=N_TOKENS,
        pad_id=PAD_ID,
        sos_id=SOS_ID,
        eos_id=EOS_ID,
        emb_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
        device=None,
    ):
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
 
        self.model = LSTMCore(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
        ).to(self.device)
 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # PAD_ID contributes nothing to the loss -> no penalty for whatever
        # gets predicted at padding positions.
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
 
    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def fit(self, training_tensors, validation_tensors, collate_fn,
            epochs=20, batch_size=32, verbose=True):
        train_loader = DataLoader(
            SQLSeqDataset(training_tensors), batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn,
        )
        history = {"training pplx": [], "validation pplx": []}
        for epoch in range(1, epochs + 1):
            train_ppl = self.run_epoch(train_loader, train=True)
            val_ppl = self.evaluate(validation_tensors, collate_fn, batch_size=batch_size)
 
            history["training pplx"].append(train_ppl)
            history["validation pplx"].append(val_ppl)
            if verbose:
                print(f"epoch {epoch:2d} | training pplx {train_ppl:7.3f} | validation pplx {val_ppl:7.3f}")
        return history
 
    def run_epoch(self, loader, train):
        self.model.train(mode=train)
        total_loss, total_tokens = 0.0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for input_ids, target_ids in loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits, _ = self.model(input_ids)             # (B, T, V)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                n_real_tokens = (target_ids != self.pad_id).sum().item()
                total_loss += loss.item() * n_real_tokens
                total_tokens += n_real_tokens
        return math.exp(total_loss / total_tokens)
 
    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def evaluate(self, sequences, collate_fn, batch_size=32):
        """Returns perplexity on a held-out list of token-id tensors."""
        loader = DataLoader(
            SQLSeqDataset(sequences), batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn,
        )
        return self.run_epoch(loader, train=False)
 
    # -----------------------------------------------------------------
    # Sampling / generation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample(self, id_to_token=None, max_len=40, temperature=1.0):
        """
        Autoregressive sampling, one token at a time. Unlike training (one
        forward call over a whole padded batch via teacher forcing),
        generation can't be batched across time: token t+1 needs the
        hidden state produced by token t, so this is a real Python loop.
 
        Returns a list of token ids, or a list of token strings if
        id_to_token is provided.
        """
        self.model.eval()
        input_id = torch.tensor([[self.sos_id]], device=self.device)
        hidden = None
        generated = [self.sos_id]
 
        for _ in range(max_len):
            logits, hidden = self.model(input_id, hidden)          # (1, 1, vocab)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)      # (1, 1)
            token = next_id.item()
            generated.append(token)
            if token == self.eos_id:
                break
            input_id = next_id  # feed only the new token; hidden carries the past
 
        if id_to_token is not None:
            return [id_to_token[i] for i in generated]
        return generated
 
    # -----------------------------------------------------------------
    # Persistence (handy once training runs get longer)
    # -----------------------------------------------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)
 
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
