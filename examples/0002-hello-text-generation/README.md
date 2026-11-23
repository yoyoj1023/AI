# 我的第一個 AI 程式：文字生成

這個小專案使用 Hugging Face Transformers 的 `pipeline`，讓文字生成模型接續一段英文提示句，並產生新的文字。

## 這個程式做了什麼？

`hello_text_generation.py` 的流程只有兩步：

1. 建立 `text-generation` pipeline；第一次執行時會下載預設文字生成模型。
2. 傳入提示句，並要求模型產生兩個序列。

## 環境需求

- Python 3.10 或更新版本
- 網路連線：第一次執行時會下載模型檔案

## 安裝與執行

在 PowerShell 中執行以下指令：

```powershell
cd C:\VScode\github\AI\examples\0002-hello-text-generation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python hello_text_generation.py
```

完成後可用以下指令離開虛擬環境：

```powershell
deactivate
```

## 程式內容

```python
from transformers import pipeline


generator = pipeline("text-generation")
result = generator(
    "In this course, we will teach you how to",
    num_return_sequences=2,
    max_length=30,
)

print(result)
```

- `num_return_sequences=2`：產生兩個不同的文字序列。
- `max_length=30`：限制每個結果包含提示句在內的總長度；實際上是 token 長度，不一定等同於空格分隔的單字數。

第一次執行可能需要較長時間。模型並不保證每次產出的內容完全相同，因此預期結果會因模型、版本與隨機設定而有所差異。
