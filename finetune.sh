export CUDA_DEVICE_ORDER=PCI_BUS_ID
export FLAGS_check_nan_inf=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#model_type="mbert"
#dataset="wiki-clir"
#batch_size=8
#
#for source_lang in "en" "es" "fr" "de"
#do
#    for target_lang in "en" "es" "fr" "de"
#    do
#        # if [ $source_lang != $target_lang ]; then
#	    sbatch group.sh $model_type $dataset $source_lang $target_lang $batch_size
#        # fi
#    done
#done python -u
# Try src,tgt = [(en,de), (de,es)] for comparision with Figure 4
fleetrun --gpus 0,1,2,3,4,5,6,7 finetune-search.py --model_type mbert \
  --model_path $1 \
  --dataset $2 \
  --source_lang $3 \
  --target_lang $4 \
  --batch_size 8 \
  --full_doc_length \
  --num_neg 1 \
  --eval_step 1 \
  --num_epochs 16 \
  --encoder_lr 1e-6 \
  --projector_lr 1e-6 \
  --num_ft_encoders 6 \
  --seed 611
