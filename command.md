conditional training
```bash
python first_sweep.py --gpu 1 --task_config /home/research/l.tingjun/diffusion/configs/train/fmri_train_lowres_cond_acc4_img256.yaml
python first_sweep.py --gpu 0 --task_config /home/research/l.tingjun/diffusion/configs/train/fmri_train_lowres_cond_acc4_img256_check.yaml
```

conditional sampling
```bash
python first_sweep.py --gpu 2 --task_config /home/research/l.tingjun/diffusion/configs/sample/fastmri_sample_cond_config.yaml
```
```bash
python first_sweep.py --gpu 1 --task_config /home/research/l.tingjun/diffusion/configs/sample/new_sample.yaml
```