# Models — Paper 1 model store

本目录管理基座模型与 `biod2` 决策模型。权重 payload 不纳入 git
（`.gitignore` 的 `Models/foundation-model/`、`Models/biod2/v*/`）；只跟踪 README 与 MANIFEST。

```
Models/
├── README.md
├── foundation-model/
│   ├── Qwen3.5-2B-Base/     # 容量低锚（hidden 2048 / 24 层）
│   ├── Qwen3.5-4B-Base/     # 容量中臂（hidden 2560 / 32 层）
│   └── Qwen3.5-9B-Base/     # 容量高臂（hidden 4096 / 32 层）
└── biod2/
    ├── MANIFEST.json        # 新 7 头/9-task 版本表；当前为空
    └── <new-version>/       # retrain 后写入：adapter + heads.pt + config + train_manifest
```

## foundation-model

三个底座都是 `.hf_cache` 的硬链接镜像（`cp -aLl`，同 inode，不额外占盘）。
2B/4B/9B 同 Qwen3.5 家族、同 tokenizer、同 `Qwen3_5ForConditionalGeneration`；
4B/9B 同为 32 层，2B 为 24 层（解释容量效应时注意深度混杂）。

## biod2

- 旧 `v1`（`biod-mt-v2` = `multitask-v5`）已归档到
  `other/archive-pre-paper1-20261103/models/biod2-v1/`；其 kb/context/relay 头不可用，不要部署。
- 新实验统一 7 头、9-task、SFT+LoRA 从零训练；2B/4B/9B 三个容量共用同一数据与 recipe。
- 新版本建议命名：`Models/biod2/paper1-2b-v1/`、`paper1-4b-v1/`、`paper1-9b-v1/`；
  训练完成后更新 `MANIFEST.json`（路径 / sha256 / tasks / data manifest / git commit / deploy pin）。

## 用法

```bash
# 基座（本地目录，不需要 HF_HOME）
--base-model Models/foundation-model/Qwen3.5-4B-Base

# 决策模型（retrain 后）
python -m biod.multitask.serve --ckpt Models/biod2/paper1-9b-v1 --host 127.0.0.1 --port 8790
```
