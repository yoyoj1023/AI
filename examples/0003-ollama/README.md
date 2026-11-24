# Ollama 快速入門

Ollama 可以在自己的電腦上下載並執行大型語言模型，提供命令列工具，也會啟動本機 HTTP 服務，讓程式透過 API 呼叫模型。模型通常在第一次使用時下載；之後可在本機執行，不必每次把提示內容送到雲端。

> 模型仍會占用磁碟空間、記憶體與運算資源。實際可執行的模型大小和速度，取決於電腦硬體及模型設定。

## 1. 安裝與第一次對話

1. 從 [Ollama 下載頁面](https://ollama.com/download)安裝適合作業系統的版本。
2. 開啟 Ollama，確認它正在執行。
3. 在終端機執行以下命令，下載並啟動範例模型：

```powershell
ollama --version
ollama run qwen3.5:4b
```

進入對話後輸入問題即可；輸入 `/bye` 離開。也可以直接指定模型和提示：

```powershell
ollama run qwen3.5:4b "用三句話介紹什麼是機器學習"
```

模型名稱會隨 Ollama 模型庫更新；可在 [模型庫](https://ollama.com/library)查找可用模型及其標籤。第一次執行 `ollama run` 會自動下載模型；也可以先用 `ollama pull qwen3.5:4b` 下載。

## 2. 常用命令列工具

| 命令 | 用途 |
| --- | --- |
| `ollama --version` | 查看 Ollama 版本 |
| `ollama pull qwen3.5:4b` | 下載模型 |
| `ollama run qwen3.5:4b` | 在終端機與模型對話 |
| `ollama ls` | 列出已下載的模型 |
| `ollama ps` | 查看目前載入中的模型 |
| `ollama stop qwen3.5:4b` | 停止執行中的模型 |
| `ollama rm qwen3.5:4b` | 刪除本機模型檔案 |
| `ollama serve` | 在終端機啟動 Ollama 服務 |

一般桌面安裝會在背景啟動服務；若 API 無法連線，可先確認 Ollama 應用程式是否開啟，再視需要執行 `ollama serve`。若服務已在執行，重複啟動可能會提示連接埠已被使用。

## 3. 本機 HTTP API

Ollama 的本機服務預設位址是 `http://localhost:11434`。原生 API 的路徑以 `/api` 開頭，例如 `http://localhost:11434/api/chat`；本機請求通常不需要 API 金鑰。呼叫前請先確認服務已啟動，而且指定模型已下載。

### 檢查服務與模型

在 PowerShell 執行：

```powershell
Invoke-WebRequest http://localhost:11434
Invoke-RestMethod http://localhost:11434/api/tags
```

第一個請求可確認服務是否有回應；第二個請求會列出本機模型。

### 文字生成：`POST /api/generate`

適合直接對一段提示生成文字：

```powershell
$body = @{
    model  = "qwen3.5:4b"
    prompt = "用一句話解釋什麼是開源軟體。"
    stream = $false
} | ConvertTo-Json

$request = @{
    Uri         = "http://localhost:11434/api/generate"
    Method      = "Post"
    ContentType = "application/json"
    Body        = $body
}
$result = Invoke-RestMethod @request

$result.response
```

### 對話生成：`POST /api/chat`

適合多輪對話。每次請求都傳入 `messages` 對話紀錄；若要延續上下文，需把先前的訊息一併傳入。

```powershell
$body = @{
    model = "qwen3.5:4b"
    messages = @(
        @{ role = "user"; content = "嗨！請用繁體中文簡短自我介紹。" }
    )
    stream = $false
} | ConvertTo-Json -Depth 10

$request = @{
    Uri         = "http://localhost:11434/api/chat"
    Method      = "Post"
    ContentType = "application/json"
    Body        = $body
}
$result = Invoke-RestMethod @request

$result.message.content
```

### 常用原生 API

| 方法與路徑 | 用途 |
| --- | --- |
| `GET /api/tags` | 列出本機模型 |
| `POST /api/pull` | 下載模型 |
| `POST /api/chat` | 對話生成 |
| `POST /api/generate` | 提示詞文字生成 |
| `POST /api/embed` | 將文字轉為向量嵌入 |
| `POST /api/show` | 查看模型資訊 |
| `DELETE /api/delete` | 刪除本機模型 |

API 預設會以串流方式回傳生成結果。如果客戶端希望一次收到完整 JSON，請在請求中加入 `"stream": false`。串流回應則是逐行傳回的 JSON 事件，需由客戶端逐段讀取。

Ollama 也提供部分 OpenAI 相容 API，基底網址為 `http://localhost:11434/v1`，例如 `/v1/chat/completions`。連接本機服務的 OpenAI SDK 通常要求提供 API key 欄位，但本機 Ollama 會忽略其值；相容範圍依 API 功能而異。

## 4. 從 Python 呼叫

Ollama 提供官方 Python 套件。先安裝：

```powershell
python -m pip install ollama
```

再執行：

```python
from ollama import chat

response = chat(
    model="qwen3.5:4b",
    messages=[{"role": "user", "content": "用繁體中文介紹 Ollama。"}],
)

print(response.message.content)
```

此範例使用本機 Ollama 服務，因此需先安裝並啟動 Ollama，且本機已有 `qwen3.5:4b` 模型。

## 5. 常見問題與安全提醒

- **找不到 `ollama` 命令**：確認已完成安裝，並重新開啟終端機；Windows 可檢查安裝程式及 PATH 設定。
- **連線被拒絕**：開啟 Ollama 桌面應用程式，或在終端機啟動 `ollama serve`。
- **模型下載或生成太慢**：模型大小、可用記憶體、上下文長度和硬體都會影響速度；可改用模型庫中較小的模型。
- **模型名稱錯誤**：使用 `ollama ls` 檢查本機已下載的名稱，並確認名稱與標籤和請求中的值相符。
- **不要直接公開本機 API**：本機服務預設不要求驗證。除非已設定適當的網路隔離與存取控制，否則不要把服務暴露到公用網路。

## 延伸閱讀

- [Ollama 官方快速開始](https://docs.ollama.com/quickstart)
- [CLI 命令參考](https://docs.ollama.com/cli)
- [HTTP API 總覽](https://docs.ollama.com/api/introduction)
- [Chat API](https://docs.ollama.com/api/chat)
- [Generate API](https://docs.ollama.com/api/generate)
- [OpenAI 相容 API](https://docs.ollama.com/api/openai-compatibility)
- [模型庫](https://ollama.com/library)
- [Ollama Python 套件](https://github.com/ollama/ollama-python)
