# 我的第一個 AI 程式：文字情緒分析

歡迎開始 AI 之旅！這個小專案會使用 Hugging Face Transformers 的 `pipeline`，判斷一段英文文字的情緒是正面還是負面。

## 這個程式做了什麼？

`hello_transformers.py` 的流程可以拆成三步：

1. 載入情緒分析的 AI 模型。
2. 把文字交給模型分析。
3. 印出模型預測的標籤和信心分數。

程式使用的 `pipeline` 是 Transformers 提供的高階介面，幫我們處理載入模型、整理輸入和執行推論等常見工作。這裡的 `sentiment-analysis` 是任務名稱；如果不指定模型，套件會選用適合此任務的預設模型。

## 環境需求

- Python 3.10 或更新版本
- 網路連線：第一次執行時會下載模型檔案

本專案使用 `.venv` 建立獨立的 Python 環境，避免套件和其他專案互相影響。

## 安裝與執行

在 PowerShell 中執行以下指令：

```powershell
cd C:\VScode\github\AI\0001-hello-transformers
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python hello_transformers.py
```

若尚未建立 `.venv`，先執行：

```powershell
python -m venv .venv
```

再回到上面的安裝與執行步驟。完成後可用以下指令離開虛擬環境：

```powershell
deactivate
```

## 預期結果

第一次執行會從 Hugging Face Hub 下載模型，所需時間依網路速度而異。完成後會看到類似結果：

```text
[{'label': 'POSITIVE', 'score': 0.9996}]
```

- `label` 是模型預測的情緒標籤，例如 `POSITIVE`（正面）或 `NEGATIVE`（負面）。
- `score` 是模型對此預測的信心分數，介於 `0` 和 `1` 之間；它不是「答案正確的保證」。

## 動手修改

打開 `hello_transformers.py`，修改傳給 `classifier` 的文字，再重新執行程式：

```python
result = classifier("I am excited to start my AI journey!")
```

試著換成其他英文句子，觀察標籤和分數如何改變。這個預設模型主要針對英文情緒分析；中文輸入可能無法得到可靠結果。

## 小小提醒

模型的輸出是根據訓練資料做出的預測，不代表它真的理解了句子的意思。諷刺、反話、文化差異和上下文都可能讓結果不準確。學習時可以把它當成觀察 AI 推論結果的起點。
