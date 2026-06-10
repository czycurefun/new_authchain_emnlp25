# One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems
This repository contains the source code for the paper One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems

# Overview
Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have shown improved performance in generating accurate responses. However, the dependence on external knowledge bases introduces potential security vulnerabilities, particularly when these knowledge bases are publicly accessible and modifiable.
Poisoning attacks on knowledge bases for RAG systems face two fundamental challenges: the injected malicious content must compete with multiple authentic documents retrieved by the retriever, and LLMs tend to trust retrieved information that aligns with their internal memorized knowledge. Previous works attempt to address these challenges by injecting multiple malicious documents, but such saturation attacks are easily detectable and impractical in real-world scenarios.
To enable the effective single document poisoning attack, we propose AuthChain, a novel knowledge poisoning attack method that leverages Chain-of-Evidence theory and authority effect to craft more convincing poisoned documents. AuthChain generates poisoned content that establishes strong evidence chains and incorporates authoritative statements, effectively overcoming the interference from both authentic documents and LLMs' internal knowledge.
Extensive experiments across six popular LLMs demonstrate that AuthChain achieves significantly higher attack success rates while maintaining superior stealthiness against RAG defense mechanisms compared to state-of-the-art baselines.

The overview of Authchain:
<p align="center">
  <img src="https://github.com/czycurefun/AuthChain/blob/main/fig/method_revise.png" width="900"/>
</p>

# Environment
```bash
pip install -r requirements.txt
```

`requirements.txt` currently only requires `requests`.



# Running
**Information Extraction**
```
python extract_information.py --input data/reproduction/sample_questions.json --output data/reproduction/sample_extract_info.mock.json --mock --overwrite --limit 1
```   

**Poisoned Document Generation**
```
python AuthChain.py --input data/reproduction/sample_extract_info.mock.json --output data/reproduction/sample_authchain.mock.json --mock --overwrite --limit 1
```   

For live runs, keep `AuthChain.py`'s `api_key` value masked and pass the real key through the environment.
The default live backend used in this local reproduction is:

- base URL: `https://antchat.alipay.com`
- model: `deepseek-v4-flash`

```bash
export ANTCHAT_API_KEY="YOUR_REAL_KEY"

python extract_information.py \
  --input data/reproduction/sample_questions.json \
  --output data/reproduction/sample_extract_info.real.json \
  --limit 1 \
  --model deepseek-v4-flash \
  --overwrite

python AuthChain.py \
  --input data/reproduction/sample_extract_info.real.json \
  --output data/reproduction/sample_authchain.real.json \
  --limit 1 \
  --model deepseek-v4-flash \
  --max-revisions 0 \
  --max-retries 5 \
  --request-sleep 70 \
  --overwrite \
  --include-diagnostics
```

`model_client.py` is only a thin local adapter around an AntChat/OpenAI-compatible Chat Completions endpoint. If you use another provider, you can replace that adapter with the provider's official SDK or official OpenAI-compatible request method while keeping `extract_information.py` and `AuthChain.py` unchanged.

See `docs/reproduction.md` for the local reproduction log and remaining experiment details.

