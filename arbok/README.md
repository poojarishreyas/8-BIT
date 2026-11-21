# Arbok tokenizer and inference server

This directory contains the renamed Arbok tokenizer and OpenAI-compatible inference server copied from the local Vajram source. It does not include the chat UI, model weights, or model training code.

## Contents

- `tokenizer/`: partitioned tokenizer implementation and vocabularies.
- `serve/server.py`: API server (`/health`, `/v1/models`, `/v1/chat/completions`).
- `serve/tools.py`, `serve/safe_http.py`, `serve/safe_python.py`: server tool support.
- `scripture/`: verified verse store needed by the server's scripture lookup tool.

Install dependencies with `pip install -r arbok/requirements.txt`, then run `python arbok/serve/server.py` from the repository root. A compatible model backend is required. The original model artifact is at https://huggingface.co/AbhiAtreyas/Vajram-KN1-8B; that URL is retained so downloads resolve.

The source license and attribution are in `LICENSE` and `NOTICE`.
