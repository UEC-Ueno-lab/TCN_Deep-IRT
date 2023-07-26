# TCN_Deep-IRT

### Run model 1
```python
!bash ./run_sh/run_nishio.sh
```

### Run model 2
```python
!python main.py --root_path /content/drive/MyDrive/Bayes_research/experiments/pytorch_models/saved_model/eight_layer1219/ \
--model_type convmem01 --epochs 200 --dataset Eddi \
--batch_size=512 --fold 1 --lr 0.001
```
