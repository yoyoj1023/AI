# Attention Is All You Need

- **標題：** Attention Is All You Need
- **作者：** Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin
- **發表年份：** 2017
- **會議：** NeurIPS 2017
- **論文：** [Attention Is All You Need.pdf](./Attention-Is-All-You-Need.pdf)
- **arXiv：** https://arxiv.org/abs/1706.03762
- **原始論文來源：** https://arxiv.org/pdf/1706.03762

## 摘要

本文提出 Transformer，一種完全基於 attention mechanisms 的序列轉換模型，捨棄 recurrent 與 convolution 架構。Transformer 由 encoder 與 decoder 的 stacked self-attention 和 point-wise fully connected layers 組成，能夠更有效地建模序列中的長距離相依關係。

在 WMT 2014 英德翻譯與英法翻譯任務上，Transformer 以較短的訓練時間達到當時的 best BLEU score；其中英德翻譯結果為 28.4 BLEU，英法翻譯結果為 41.8 BLEU。模型也被證明可以有效泛化到其他工作，例如英語成分句法分析。

## 核心貢獻

- 提出只使用 attention、完全移除 recurrence 與 convolution 的 Transformer 架構。
- 引入 scaled dot-product attention，提升 attention 計算的穩定性與效率。
- 使用 multi-head attention，讓模型能從不同表示子空間學習多種類型的相依關係。
- 引入 positional encoding，為純 attention 模型補充序列的順序資訊。
- 透過並行化縮短訓練時間，同時保有高品質的序列轉換效果。

## 架構重點

### Encoder

- 包含 6 個相同的 encoder layer。
- 每層包含 multi-head self-attention 與 position-wise feed-forward network。
- 同時使用 residual connection、layer normalization 與 dropout。

### Decoder

- 包含 6 個相同的 decoder layer。
- 除 encoder-decoder attention 外，也加入 masked multi-head self-attention，確保模型無法看到未來的 token。
- Encoder 輸出會提供每個 decoder position 的來源序列資訊。

## 論文引用

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```
