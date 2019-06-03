# Introduction <img src="fairseq_logo.png" width="50"> 

-**Personal change:**
  *add a new model named len_pre_transformer.py. it takes advantages of transfomer's encoder and get rid of the decoder. as for predict sentence length, it stacks three fully connected layers above the encoder, with a 'mean' operation between the first two layers.
  *add a new crition named acc_label_smooth.py. it add acc model in the crition.
  *add 'train_acc' and 'valid_acc' in train.py and trainer.py. by doing this




-**Usage:**

-**data-prapare:**
'''
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..
'''
 Binarize the dataset:
 '''
$ TEXT=examples/translation/iwslt14.tokenized.de-en
$ fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en
 '''
-**train**
'''
CUDA_VISIBLE_DEVICES=0 fairseq-train ../../simple_transformer/data-bin/iwslt14.tokenized.de-en \
  -a group_transformer --optimizer adam --lr 0.0005 -s de -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000  --task translation\
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion group_transformer_entropy --max-epoch 88 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --fp16 --length-pre-dim 30 --save-dir len_pre_checkpoints --tensorboard-logdir ./len_pre_tensorlog --reset-optimizer>r_train.txt
  '''
  
-**inference**
'''
fairseq-generate ../../simple_transformer/data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/checkpoint_last.pt \
  --batch-size 128 --beam 5 >r_generate.txt
  '''
