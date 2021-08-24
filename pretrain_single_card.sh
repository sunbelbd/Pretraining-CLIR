export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7
export FLAGS_check_nan_inf=True
# need to calculate epoch_size (epoch_size's unit is number of positive qd pair) very carefully
# e.g., for each language pair with each task, # pos-qd-pairs / epoch = epoch_size * NGPU * batch_size

python -u train.py --exp_name test \
  --fp16 false \
  --model_type mbert \
  --model_path bert-base-multilingual-uncased \
  --batch_size 8 \
  --qlm_mask_mode query \
  --mlm_probability 0.25 \
  --optimizer adam,lr=0.000005 \
  --clip_grad_norm 5 \
  --epoch_size 8000 \
  --max_epoch 10 \
  --accumulate_gradients 1 \
  --lambda_qlm 1 \
  --lambda_rr 1 \
  --num_neg 2 \
  --max_pairs 100000 \
  --qlm_steps enfr,enes,ende,fres,frde,esde \
  --rr_steps enfr,enes,ende,fres,frde,esde
