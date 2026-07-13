from autoregression.data.dataprep import N_TOKENS, PAD_ID, EOS_ID, SOS_ID, TOKEN_TO_ID, ID_TO_TOKEN, collate
from autoregression.data.dataprep import SQLSeqDataset
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TransformerCore(nn.Module):
    """
    Token embedding + learned positional embedding -> stack of causal
    self-attention encoder layers -> Linear projection to vocab logits.

    This is a decoder-only (GPT-style) setup: built from nn.TransformerEncoderLayer,
    but with a causal mask so position t can only attend to positions <= t.
    That's what makes it valid for next-token prediction -- without the mask,
    a position could "see" its own future target through attention, which
    would make the whole teacher-forcing setup meaningless (the model could
    just copy the answer instead of predicting it).
    """

    def __init__(self, vocab_size, emb_dim, num_heads, num_layers,
                 ff_dim, dropout, pad_id, max_len):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.emb_dim = emb_dim

        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN, more stable to train than post-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, input_ids):
        """
        input_ids: (B, T) LongTensor
        returns logits: (B, T, vocab_size)

        Unlike the LSTM, there's no recurrent `hidden` state to thread
        between calls -- every position attends directly to every earlier
        position via attention, so the "memory" is just the sequence
        itself, recomputed on every call. That's also why generation
        below re-feeds the whole prefix each step instead of carrying
        state forward (see the KV-cache note in `sample`).
        """
        B, T = input_ids.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)            # (B, T, emb_dim)
        x = self.dropout(x)

        # Causal mask: position i may only attend to positions <= i.
        # True/float('-inf') entries are the positions that get masked out.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(input_ids.device)

        # Padding mask: True marks PAD positions, which should be ignored
        # as *keys* by every query (a pad token shouldn't influence anyone's
        # attention output, even though it's still a position in the tensor).
        key_padding_mask = (input_ids == self.pad_id)  # (B, T)

        out = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        out = self.ln_f(out)
        logits = self.head(out)  # (B, T, vocab)
        return logits


class TransformerAgent:
    """
    Owns the Transformer model plus everything needed to train, sample from,
    and evaluate it on the SQL-token sequences. Same interface as RNNAgent,
    so it's a drop-in swap for comparison.
    Usage:
        agent = TransformerAgent(vocab_size=N_TOKENS, pad_id=PAD_ID,
                                  sos_id=SOS_ID, eos_id=EOS_ID, max_len=64)
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
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        max_len=64,
        lr=3e-4,
        device=None,
    ):
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TransformerCore(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_id=pad_id,
            max_len=max_len,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
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

                logits = self.model(input_ids)             # (B, T, V) -- no hidden state to unpack
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )
                if train:
                    loss.backward()
                    # Transformers are less prone to exploding gradients than
                    # RNNs (no repeated multiplication across timesteps), but
                    # clipping is still cheap insurance, especially early in
                    # training with a from-scratch model.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                n_real_tokens = (target_ids != self.pad_id).sum().item()
                total_loss += loss.item() * n_real_tokens
                total_tokens += n_real_tokens
        return math.exp(total_loss / total_tokens)

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def evaluate(self, sequences, collate_fn, batch_size=32, n=1, verbose=False):
        """Returns perplexity on a held-out list of token-id tensors."""
        loader = DataLoader(
            SQLSeqDataset(sequences), batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn,
        )
        pplx_lst = []
        for _ in range(0, n):
            pplx = self.run_epoch(loader, train=False)
            if n == 1:
                return pplx
            else:
                pplx_lst.append(pplx)
        avg_pplx = sum(pplx_lst) / len(pplx_lst)
        if verbose:
            print(f"Average perplexity over {n} runs: {round(avg_pplx,2)}")
        return avg_pplx

    # -----------------------------------------------------------------
    # Sampling / generation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample(self, id_to_token=None, max_len=40, temperature=1.0):
        """
        Autoregressive sampling. Note the difference from the LSTM's
        sample(): there's no persistent `hidden` to carry forward, so
        each step re-runs the model on the *entire prefix so far*
        (re-masked causally each time), not just the newest token.

        This is the classic quadratic-cost tradeoff of naive Transformer
        generation -- step t costs O(t) instead of O(1). A proper
        implementation would cache each layer's key/value projections
        for already-seen positions and only compute attention for the
        new token (KV-caching), which is what makes production LLM
        inference fast. Left out here for clarity; worth adding once
        you're generating longer sequences.
        """
        self.model.eval()
        generated = [self.sos_id]

        for _ in range(max_len):
            input_ids = torch.tensor([generated[-self.max_len:]], device=self.device)  # (1, T)
            logits = self.model(input_ids)                       # (1, T, vocab)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)     # (1, 1)
            token = next_id.item()
            generated.append(token)
            if token == self.eos_id:
                break

        if id_to_token is not None:
            return [id_to_token[i] for i in generated]
        return generated

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))