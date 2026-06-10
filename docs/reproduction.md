# AuthChain 复现用法

## 1. 准备

进入仓库：

```bash
cd /home/public/minzhi/AuthChain
```

安装依赖：

```bash
pip install -r requirements.txt
```

配置 API key，推荐使用环境变量，不要把真实 key 写进代码：

```bash
export ANTCHAT_API_KEY="YOUR_REAL_KEY"
```

`AuthChain.py` 里的 `api_key` 只保留脱敏占位，例如：

```python
api_key = "XXXXX"
```

脚本会忽略带 `*` 的脱敏占位。如果要换 key，直接重新 `export ANTCHAT_API_KEY=...` 即可。

默认模型调用配置：

- API base URL：`https://antchat.alipay.com`
- Model：`deepseek-v4-flash`
- 调用封装：`model_client.py`

`model_client.py` 是统一的模型调用适配层。`extract_information.py` 和 `AuthChain.py` 只调用：

```python
llm.complete(prompt)
```

所以后续换成 DeepSeek 官方 API、Qwen 官方 API、OpenAI-compatible endpoint 或服务商官方 SDK 时，只需要替换 `model_client.py` 或传入新的 `--api-base-url`、`--model`、`--api-key-env`。

## 2. 一键跑通

运行：

```bash
./run_reproduction.sh
```

该脚本会跑 1 条样本：

1. 抽取 `Intent / Entities / Relations`
2. 生成 AuthChain 文档

输出文件：

- `data/reproduction/sample_extract_info.real.json`
- `data/reproduction/sample_authchain.real.json`

## 3. 手动运行

信息抽取：

```bash
python3 extract_information.py \
  --input data/reproduction/sample_questions.json \
  --output data/reproduction/sample_extract_info.real.json \
  --overwrite \
  --limit 1 \
  --model deepseek-v4-flash
```

AuthChain 文档生成：

```bash
python3 AuthChain.py \
  --input data/reproduction/sample_extract_info.real.json \
  --output data/reproduction/sample_authchain.real.json \
  --overwrite \
  --limit 1 \
  --model deepseek-v4-flash \
  --include-diagnostics \
  --max-revisions 0 \
  --max-retries 5 \
  --request-sleep 70
```

## 4. 离线检查

不调用 API，只检查本地流程：

```bash
python3 extract_information.py \
  --input data/reproduction/sample_questions.json \
  --output data/reproduction/sample_extract_info.mock.json \
  --limit 1 \
  --mock \
  --overwrite

python3 AuthChain.py \
  --input data/reproduction/sample_extract_info.mock.json \
  --output data/reproduction/sample_authchain.mock.json \
  --limit 1 \
  --mock \
  --overwrite \
  --include-diagnostics
```

## 5. DeepSeek 官方 API 跑法

DeepSeek 官方文档给出的 OpenAI-compatible base URL 是：

```text
https://api.deepseek.com
```

官方文档示例使用 `DEEPSEEK_API_KEY` 环境变量和 OpenAI SDK；本仓库当前的 `model_client.py` 也支持直接用 `requests` 调这个官方 endpoint。

### 5.1 不改代码，直接跑 DeepSeek 官方 endpoint

准备 key：

```bash
export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_KEY"
```

信息抽取：

```bash
python3 extract_information.py \
  --input data/reproduction/sample_questions.json \
  --output data/reproduction/sample_extract_info.deepseek.json \
  --limit 1 \
  --model deepseek-v4-flash \
  --api-base-url https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --overwrite
```

AuthChain 文档生成：

```bash
python3 AuthChain.py \
  --input data/reproduction/sample_extract_info.deepseek.json \
  --output data/reproduction/sample_authchain.deepseek.json \
  --limit 1 \
  --model deepseek-v4-flash \
  --api-base-url https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --overwrite \
  --include-diagnostics \
  --max-revisions 0 \
  --max-retries 5 \
  --request-sleep 10
```

### 5.2 使用 DeepSeek 官方 SDK 风格代码

如果希望调用代码和 DeepSeek 官方文档里的 Python 示例保持一致，先安装 OpenAI SDK：

```bash
pip install openai
```

然后把示例适配器复制为当前调用层：

```bash
cp examples/model_client_deepseek_official.py model_client.py
```

之后仍然用上面的 `DEEPSEEK_API_KEY` 命令运行即可。

示例适配器的位置：

```text
examples/model_client_deepseek_official.py
```

这个文件保留了仓库需要的接口：

```python
class RemoteChatClient:
    def complete(self, prompt: str, **kwargs) -> str:
        ...
```

内部使用 DeepSeek 官方文档同类写法：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": prompt},
    ],
    stream=False,
)

text = response.choices[0].message.content
```

实际仓库适配器里额外处理了重试、`system_prompt`、`reasoning_content` 兜底和 `request_sleep`。

## 6. 其他 OpenAI-compatible 接口

如果使用 Qwen、OpenAI-compatible 网关或其他服务商 endpoint：

```bash
export PROVIDER_API_KEY="YOUR_REAL_KEY"

python3 extract_information.py \
  --input data/reproduction/sample_questions.json \
  --output data/reproduction/sample_extract_info.real.json \
  --limit 1 \
  --model "MODEL_ID" \
  --api-base-url "OPENAI_COMPATIBLE_BASE_URL" \
  --api-key-env PROVIDER_API_KEY \
  --overwrite

python3 AuthChain.py \
  --input data/reproduction/sample_extract_info.real.json \
  --output data/reproduction/sample_authchain.real.json \
  --limit 1 \
  --model "MODEL_ID" \
  --api-base-url "OPENAI_COMPATIBLE_BASE_URL" \
  --api-key-env PROVIDER_API_KEY \
  --overwrite \
  --include-diagnostics
```

如果服务商提供官方 SDK，建议在 `model_client.py` 里适配 SDK 调用，并保持 `RemoteChatClient.complete(prompt, **kwargs) -> str` 这个接口不变。
