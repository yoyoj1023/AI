# 用 PyTorch 實作中英 Transformer 翻譯

這個練習從零組出一個小型 Encoder-Decoder Transformer，不使用預訓練模型或翻譯套件。程式用一份迷你中英句對資料訓練同一個模型，並以方向標記支援中翻英、英翻中。

> 這是用來理解架構的教學範例，不是實用翻譯器。訓練資料非常少，模型主要能記住示範句；要提升翻譯品質需要大量且多樣的平行語料。

## 環境與執行

需要 Python 3.10 或更新版本，以及 PyTorch。建議在此資料夾建立獨立的虛擬環境：

```powershell
cd C:\VScode\github\AI\examples\0003-pytorch-transformer-nmt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python transformer_nmt.py
```

程式會在小型資料集上訓練模型，再印出中翻英與英翻中的示範。若想自行輸入句子：

```powershell
python transformer_nmt.py --interactive
```

可以用 `--epochs 400` 增加訓練輪數，或用 `--seed 7` 更換隨機種子。程式會優先使用 CUDA，否則使用 CPU。

## 架構對照

```text
來源句 + 方向標記
        │
        ▼
Token Embedding + Positional Encoding
        │
        ▼
Encoder Self-Attention ── Padding Mask
        │
        │ encoder memory
        ▼
Decoder Masked Self-Attention ── Causal Mask + Padding Mask
        │
        ▼
Cross-Attention ◀────────────── encoder memory + source Padding Mask
        │
        ▼
Feed-Forward → Linear → 下一個 token
```

- **Encoder**：讀取完整來源句，因此 Encoder Self-Attention 不使用因果遮罩；但會用 Padding 遮罩忽略批次補齊的位置。
- **Decoder 的因果遮罩（Causal Mask）**：遮住目前位置右側的未來 token。訓練時可以一次輸入整段目標句，同時避免模型偷看答案。
- **Padding 遮罩（Padding Mask）**：忽略為了把不同長度句子組成 batch 而補上的 `<pad>`。Decoder Self-Attention 會忽略目標端 padding，Cross-Attention 會忽略來源端 padding。
- **Cross-Attention**：Query 來自 Decoder；Key 和 Value 來自 Encoder。Decoder 因而能在產生每個目標 token 時，參照來源句的內容。

## 訓練時的右移目標句

假設目標句 token 是 `hello .`，訓練會把 Decoder 輸入和預期答案錯開一格：

```text
Decoder 輸入：<bos> hello
預期答案：   hello  . <eos>
```

`<bos>` 表示開始、`<eos>` 表示結束。推論時則從 `<bos>` 開始，每次把剛預測的 token 接回輸入，直到模型輸出 `<eos>`。

## 試著動手改

在 `transformer_nmt.py` 的 `PAIRS` 新增中英文句對，重新執行觀察模型輸出。資料量很小時，訓練和示範輸出都不代表真實語言能力；特別是未出現在訓練資料中的句子，結果通常不可靠。
