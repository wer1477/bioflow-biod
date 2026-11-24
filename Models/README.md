# Models

Model payloads are not tracked in this repository.

## Base tower

The released checkpoint is trained on `Qwen/Qwen3.5-4B-Base`. Download the base tower from Hugging
Face and either place it at the absolute path recorded in the checkpoint's `model_config.json`, or edit
that entry to point at your local copy. The base model is not redistributed here and remains under the
Qwen model license.

## Released checkpoint

The evaluated five-head checkpoint is released at
[huggingface.co/wer1477/biod-mt-v2-4b](https://huggingface.co/wer1477/biod-mt-v2-4b):

```bash
scripts/download-weights.sh /path/to/biod-mt-v2-4b
```

The download fetches exactly five files, so the directory fingerprint reported by
`/v1/health` (`sha256-tree-v1`) matches the checkpoint hash in the paper:

```text
biod-mt-v2-4b/
├── heads.pt
├── model_config.json
└── adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── README.md
```

If you download the Hugging Face repository with a generic client, select these five paths explicitly;
extra files in the directory change the directory fingerprint.

## Local layout

```text
Models/
├── foundation-model/
│   └── Qwen3.5-4B-Base/   # not tracked
└── biod-mt-v2-4b/         # not tracked; released on Hugging Face
```
