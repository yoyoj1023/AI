import argparse
import re
import random
import sys

import torch
from torch import nn


PAD, BOS, EOS, UNK, TO_EN, TO_ZH = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<2en>",
    "<2zh>",
)

PAIRS = [
    ("你好。", "hello."),
    ("早上好。", "good morning."),
    ("晚上好。", "good evening."),
    ("謝謝你。", "thank you."),
    ("非常感謝。", "thank you very much."),
    ("再見。", "goodbye."),
    ("你好吗？", "how are you?"),
    ("我很好。", "i am fine."),
    ("我愛學習。", "i love learning."),
    ("今天天氣很好。", "the weather is nice today."),
    ("我喜歡喝咖啡。", "i like drinking coffee."),
    ("請幫幫我。", "please help me."),
]


def tokenize(text):
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]", text.lower())


def is_chinese(text):
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def detokenize(tokens, chinese):
    if chinese:
        return "".join(tokens)
    text = " ".join(tokens)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


class Vocabulary:
    def __init__(self, examples):
        tokens = {PAD, BOS, EOS, UNK, TO_EN, TO_ZH}
        for source, target, _ in examples:
            tokens.update(tokenize(source))
            tokens.update(tokenize(target))
        self.tokens = [PAD, BOS, EOS, UNK, TO_EN, TO_ZH] + sorted(
            tokens - {PAD, BOS, EOS, UNK, TO_EN, TO_ZH}
        )
        self.to_id = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id = self.to_id[PAD]
        self.bos_id = self.to_id[BOS]
        self.eos_id = self.to_id[EOS]
        self.unk_id = self.to_id[UNK]

    def encode(self, tokens):
        return [self.to_id.get(token, self.unk_id) for token in tokens]


class PositionalEncoding(nn.Module):
    def __init__(self, dimension, max_length=128):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2) * (-torch.log(torch.tensor(10000.0)) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * scale)
        encoding[:, 1::2] = torch.cos(positions * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, tokens):
        return self.encoding[:, : tokens.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, dimension, heads, feedforward_dimension, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension),
            nn.ReLU(),
            nn.Linear(feedforward_dimension, dimension),
        )
        self.norm_attention = nn.LayerNorm(dimension)
        self.norm_feedforward = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, padding_mask):
        attended = self.attention(
            values, values, values, key_padding_mask=padding_mask, need_weights=False
        )[0]
        values = self.norm_attention(values + self.dropout(attended))
        return self.norm_feedforward(values + self.dropout(self.feedforward(values)))


class DecoderLayer(nn.Module):
    def __init__(self, dimension, heads, feedforward_dimension, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension),
            nn.ReLU(),
            nn.Linear(feedforward_dimension, dimension),
        )
        self.norm_self = nn.LayerNorm(dimension)
        self.norm_cross = nn.LayerNorm(dimension)
        self.norm_feedforward = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, memory, causal_mask, target_padding_mask, source_padding_mask):
        attended = self.self_attention(
            values,
            values,
            values,
            attn_mask=causal_mask,
            key_padding_mask=target_padding_mask,
            need_weights=False,
        )[0]
        values = self.norm_self(values + self.dropout(attended))

        # Cross-attention queries come from the decoder; keys and values come from the encoder.
        attended = self.cross_attention(
            values, memory, memory, key_padding_mask=source_padding_mask, need_weights=False
        )[0]
        values = self.norm_cross(values + self.dropout(attended))
        return self.norm_feedforward(values + self.dropout(self.feedforward(values)))


class TransformerNMT(nn.Module):
    def __init__(self, vocabulary_size, pad_id, dimension=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocabulary_size, dimension, padding_idx=pad_id)
        self.position = PositionalEncoding(dimension)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(dimension, heads, dimension * 4, dropout) for _ in range(layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(dimension, heads, dimension * 4, dropout) for _ in range(layers)
        )
        self.output = nn.Linear(dimension, vocabulary_size)

    def _embed(self, tokens):
        return self.dropout(self.embedding(tokens) * (self.embedding.embedding_dim**0.5) + self.position(tokens))

    def encode(self, source_tokens):
        source_padding_mask = source_tokens.eq(self.pad_id)
        values = self._embed(source_tokens)
        for layer in self.encoder_layers:
            values = layer(values, source_padding_mask)
        return values, source_padding_mask

    def decode(self, target_tokens, memory, source_padding_mask):
        target_padding_mask = target_tokens.eq(self.pad_id)
        length = target_tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=target_tokens.device), diagonal=1
        )
        values = self._embed(target_tokens)
        for layer in self.decoder_layers:
            values = layer(
                values, memory, causal_mask, target_padding_mask, source_padding_mask
            )
        return self.output(values)

    def forward(self, source_tokens, target_tokens):
        memory, source_padding_mask = self.encode(source_tokens)
        return self.decode(target_tokens, memory, source_padding_mask)


def make_training_data(vocabulary):
    examples = []
    for chinese, english in PAIRS:
        examples.append((chinese, english, TO_EN))
        examples.append((english, chinese, TO_ZH))

    source_batch, target_input_batch, target_output_batch = [], [], []
    for source, target, direction in examples:
        source_ids = vocabulary.encode([direction] + tokenize(source))
        target_ids = [vocabulary.bos_id] + vocabulary.encode(tokenize(target)) + [
            vocabulary.eos_id
        ]
        source_batch.append(source_ids)
        target_input_batch.append(target_ids[:-1])
        target_output_batch.append(target_ids[1:])

    def pad_sequences(sequences, pad_id):
        width = max(map(len, sequences))
        return torch.tensor(
            [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences],
            dtype=torch.long,
        )

    return (
        pad_sequences(source_batch, vocabulary.pad_id),
        pad_sequences(target_input_batch, vocabulary.pad_id),
        pad_sequences(target_output_batch, vocabulary.pad_id),
    )


def train(model, vocabulary, epochs, device):
    sources, targets, labels = make_training_data(vocabulary)
    sources, targets, labels = sources.to(device), targets.to(device), labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(sources, targets)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"epoch {epoch:>3}/{epochs} | loss {loss.item():.4f}")


def translate(model, vocabulary, sentence, device, max_length=32):
    chinese_source = is_chinese(sentence)
    direction = TO_EN if chinese_source else TO_ZH
    source_tokens = torch.tensor(
        [vocabulary.encode([direction] + tokenize(sentence))], dtype=torch.long, device=device
    )

    model.eval()
    with torch.no_grad():
        memory, source_padding_mask = model.encode(source_tokens)
        generated = torch.tensor([[vocabulary.bos_id]], dtype=torch.long, device=device)
        forbidden = [
            vocabulary.to_id[token] for token in (PAD, BOS, UNK, TO_EN, TO_ZH)
        ]
        for _ in range(max_length):
            logits = model.decode(generated, memory, source_padding_mask)[:, -1, :]
            logits[:, forbidden] = float("-inf")
            next_token = logits.argmax(dim=-1, keepdim=True)
            if next_token.item() == vocabulary.eos_id:
                break
            generated = torch.cat((generated, next_token), dim=1)

    output_tokens = [
        vocabulary.tokens[index]
        for index in generated[0, 1:].tolist()
        if vocabulary.tokens[index] not in {PAD, BOS, EOS, UNK, TO_EN, TO_ZH}
    ]
    return detokenize(output_tokens, chinese=not chinese_source)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="用 PyTorch 從零訓練一個迷你中英 Transformer。")
    parser.add_argument("--epochs", type=int, default=250, help="訓練輪數，預設 250")
    parser.add_argument("--interactive", action="store_true", help="訓練後進入互動翻譯")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必須大於 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = [(zh, en, TO_EN) for zh, en in PAIRS]
    vocabulary = Vocabulary(examples)
    model = TransformerNMT(len(vocabulary.tokens), vocabulary.pad_id).to(device)

    print(f"使用裝置：{device}")
    print(f"訓練資料：{len(PAIRS) * 2} 組方向（雙向共用一個模型）")
    train(model, vocabulary, args.epochs, device)

    print("\n翻譯示範：")
    for sentence in ("你好。", "我喜歡喝咖啡。", "good morning.", "how are you?"):
        print(f"  {sentence} -> {translate(model, vocabulary, sentence, device)}")

    if args.interactive:
        print("\n輸入中文或英文開始翻譯，輸入 q 離開。")
        while True:
            sentence = input("> ").strip()
            if sentence.lower() == "q":
                break
            if sentence:
                print(translate(model, vocabulary, sentence, device))


if __name__ == "__main__":
    main()
