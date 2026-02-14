# CEAT + SFDA  Skeleton

This repo provides a runnable skeleton for:
- Multimodal encoders (EEG/Audio/Text)
- CEAT-style contrastive alignment (E-A, E-T, A-T)
- Source training (supervised)
- Target SFDA adaptation 
- Evaluation

## Quick start (synthetic data)
```bash
pip install -r requirements.txt


python scripts/_debug_data.py

python scripts/_debug_model.py

python scripts/train_source.py --data_cfg configs/seed.yaml --model_cfg configs/model.yaml

python scripts/adapt_target_sfda.py --data_cfg configs/seed.yaml --model_cfg configs/model.yaml --sfda_cfg configs/sfda.yaml \
  --source_ckpt runs/seed_emotion/source_best.pt

python scripts/evaluate.py --data_cfg configs/seed.yaml --model_cfg configs/model.yaml --ckpt runs/seed_emotion/source_best.pt
