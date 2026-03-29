# UrbanSTCL — Downstream Tasks

This folder contains downstream-task code used with UrbanSTCL-style pretrained features.

本目录是“下游任务集合”，每个任务一般都有独立的 `requirements.txt` / `README.md`。

## Tasks

### Place Recognition (Visual Place Recognition)

- Code: `place_recognition/`
- Docs: `place_recognition/README.md`
- Install:

```bash
pip install -r place_recognition/requirements.txt
```

If you already have embeddings extracted by `../pretrain/main_test_gsv1m_feature.py`, you can plug them into the downstream pipelines depending on the task scripts/notebooks.

