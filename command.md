conditional training
```bash
python first_sweep.py --gpu 1 --task_config /home/research/l.tingjun/diffusion_mentoring/configs/train/fmri_train_lowres_cond_acc4_img256.yaml
```

conditional sampling
```bash
python first_sweep.py --gpu 2 --task_config /home/research/l.tingjun/diffusion_mentoring/configs/sample/fastmri_sample_cond_config.yaml
```