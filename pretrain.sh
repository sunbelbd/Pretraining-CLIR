export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
export FLAGS_check_nan_inf=True
# need to calculate epoch_size (epoch_size's unit is number of positive qd pair) very carefully
# e.g., for each language pair with each task, # pos-qd-pairs / epoch = epoch_size * NGPU * batch_size
fleetrun --gpus 4,5,6,7 --log_dir logs/paddle_pretrain_multi/ train.py --exp_name xlir-pretrain \
  --fp16 False \
  --multi_gpu True \
  --model_type mbert \
  --model_path bert-base-multilingual-uncased \
  --batch_size 2 \
  --qlm_mask_mode query \
  --mlm_probability 0.30 \
  --optimizer adam,lr=0.00002 \
  --clip_grad_norm 5 \
  --epoch_size 2000 \
  --max_epoch 10 \
  --accumulate_gradients 1 \
  --lambda_qlm 1 \
  --lambda_rr 1 \
  --num_neg 1 \
  --max_pairs 100000 \
  --qlm_steps enfr,enes,ende,fres,frde,esde \
  --rr_steps enfr,enes,ende,fres,frde,esde
