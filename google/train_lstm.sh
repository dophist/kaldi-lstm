#!/bin/bash

# Copyright 2012/2013  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts
train_tool=bin/bd-nnet-train-lstm-streams

# NETWORK INITIALIZATION
mlp_init=          # select initialized MLP (override initialization)
feature_transform=
num_hidden_layers=3
add_layers_period=1
halving=0
halving_count=0
iter=0
pretrain_first=1
data_test=
bin_dir=./bin
left_context=4
right_context=4
update_masks=
#
init_opts=         # options, passed to the initialization script

# FEATURE PROCESSING
# feature config (applies always)
norm_vars=true # use variance normalization?
delta_order=2

# TRAINING SCHEDULER
learn_rate=0.0001
momentum=0.9
batch_size=20  ## !!!
num_stream=4  ## !!!
targets_delay=5  ## !!!
l1_penalty=0
l2_penalty=0
# data processing
minibatch_size=256
randomizer_size=5500000
randomizer_seed=777
# learn rate scheduling
max_iters=50
min_iters=
#start_halving_inc=0.5
#end_halving_inc=0.1
start_halving_impr=0.001
end_halving_impr=0.001
halving_factor=0.5
# misc.
verbose=3
# tool
# OTHER
use_gpu_id= # manually select GPU id to run on, (-1 disables GPU)

acwt=0.10 # note: only really affects pruning (scoring is on lattices).
beam=13.0
latbeam=8.0
min_active=200
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes

skip_scoring=false
scoring_opts="--min-lmwt 4 --max-lmwt 15"

num_threads=10 # if >1, will use latgen-faster-parallel
use_gpu="yes" # yes|no|optionaly

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. utils/parse_options.sh || exit 1;


if [ $# != 4 ]; then
   #echo "Usage: $0 <data-train> <lang-dir> <ali-train> <exp-dir> <data-test>"
   echo "Usage: $0 <data-train> <lang-dir> <ali-train> <exp-dir>"
   echo " e.g.: $0 data/train data/lang exp/mono_ali exp/mono_nnet"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4
data_test=$5

#train_tool=nnet-train-perutt   # jiayu: rnn needs utterance level training

#for f in $alidir/final.mdl $alidir/ali.1.gz; do
#  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
#done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"

mkdir -p $dir/{log,nnet}

num_utts_subset=10000

##### PREPARE ALIGNMENTS ######
if [ ! -f $dir/valid_uttlist ];then
    awk '{print $1}' $data/feats.scp | utils/shuffle_list.pl | head -$num_utts_subset > $dir/valid_uttlist || exit 1;
fi

# define pdf-alignment rspecifiers
labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |" # for analyze-counts.

## for baidu SP 
#labels_tr="ark:baidu-ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
#labels_cv="ark:baidu-ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
#labels_tr_pdf="ark:baidu-ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |" # for analyze-counts.

#labels_tr_phn="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"

# get pdf-counts, used later to post-process DNN posteriors
#analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1
# copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1
# copy the tree
cp $alidir/tree $dir/tree || exit 1

## make phone counts for analysis
#analyze-counts --verbose=1 --symbol-table=$lang/phones.txt "$labels_tr_phn" /dev/null 2>$dir/log/analyze_counts_phones.log || exit 1

#feats_tr="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp | shuf | apply-cmvn --norm-vars=$norm_vars $data/norm_global_mv.ark scp:- ark:- | splice-feats --left-context=$left_context --right-context=$right_context ark:- ark:-|"
#feats_cv="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=$norm_vars $data/norm_global_mv.ark scp:- ark:- | splice-feats --left-context=$left_context --right-context=$right_context ark:- ark:-|"

feats_tr="scp:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp| shuf|"
feats_cv="scp:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp|"

## dataset 178
## for DNN
#feats_tr="ark:splice-feats --left-context=5 --right-context=5 ark:data/KaldiFeature/1-10000.ark ark:- |"
#feats_cv="ark:splice-feats --left-context=5 --right-context=5 ark:data/KaldiFeature/60001-70000.ark ark:- |"
#
##for LSTM
#feats_tr="scp:shuf data/KaldiFeature/train.scp |"
#feats_cv="ark:data/KaldiFeature/60001-70000.ark"
#
#labels_tr="ark,t:data/KaldiAlignment/target.post.ark.txt"
#labels_cv="ark,t:data/KaldiAlignment/target.post.ark.txt"

##### INITIALIZE THE NNET ######

echo 
echo "# NN-INITIALIZATION"
if [ -z "$mlp_init" ]; then
    mlp_proto=$dir/nnet.proto
    mlp_init=$dir/nnet.init
    log=$dir/log/nnet_initialize.log
    $bin_dir/nnet-initialize $mlp_proto $mlp_init 2>$log || { cat $log; exit 1; } 
fi

###### TRAIN ######
echo
echo "# RUNING LAYER-BP TRAINING"

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

##############################
#start training

# choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
mlp_base=nnet

## cross-validation on original network
#log=$dir/log/iter00.initial.log; hostname>$log
#$train_tool --feature-transform=$feature_transform --cross-validate=true \
# --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --verbose=$verbose \
# "$feats_cv" "$labels_cv" $mlp_best \
# 2>> $log || exit 1;

#loss=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
#loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
#echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"

# training
while [ $iter -lt $max_iters ]; do
    
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  seed=`date +%s`
##  if [ $iter -ge $pretrain_first ] && \
##      [ $iter -lt $[($num_hidden_layers-1)*$add_layers_period+$pretrain_first] ] && \
##      [ $[($iter-$pretrain_first%$add_layers_period) % $add_layers_period] -eq 0 ]; then
##      mlp_best="$bin_dir/nnet-initialize --seed=$seed $dir/hidden.conf - | $bin_dir/bd-nnet-insert $mlp_best - - |"
##  fi

  # training
  log=$dir/log/iter${iter}.tr.log; hostname>$log
  $train_tool --feature-transform=$feature_transform \
   --batch-size=$batch_size \
   --num-stream=$num_stream \
   --targets-delay=$targets_delay \
   --use-gpu=yes \
   --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
   --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true --verbose=$verbose \
   --binary=true \
   --randomizer-seed=$seed \
   "$feats_tr" "$labels_tr" "$mlp_best" $mlp_next \
   2>> $log || exit 1; 
  tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "
  
  # cross-validation
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  $train_tool --feature-transform=$feature_transform --cross-validate=true \
   --use-gpu=yes \
   --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --verbose=$verbose \
   "$feats_cv" "$labels_cv" $mlp_next \
   2>>$log || exit 1;
  
  loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

  # accept or reject new parameters (based on objective function)
  loss_prev=$loss
  mlp_test=
  if [ $iter -lt $[($num_hidden_layers-1)*$add_layers_period+$pretrain_first] ]; then
      loss=$loss_new
      mlp_best=$mlp_next
      echo -n "nnet accepted, TEST "
      mlp_test=$mlp_best
  else
      if [ "1" == "$(awk "BEGIN{print($loss_new<$loss);}")" ]; then
          loss=$loss_new
          mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
          mv $mlp_next $mlp_best
          mlp_test=$mlp_best
          echo -n "nnet accepted, TEST "
      else
          mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
          mv $mlp_next $mlp_reject
          mlp_test=$mlp_reject
          echo -n "nnet rejected, TEST "
      fi
      # stopping criterion
      halving=0
      # start annealing when improvement is low
      if [ "1" == "$(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev < $start_halving_impr)}")" ]; then
          halving=1
          halving_count=$((halving_count+1))
      fi
      if [ "1" == "$halving" ]; then
          learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
      fi
  fi
  
#  #test
#  class_frame_counts=$dir/ali_train_pdf.counts
#  feats_test="ark,s,cs:apply-cmvn --norm-vars=true $data/norm_global_mv.ark scp:$data_test/feats.scp ark:-| splice-feats --left-context=$left_context --right-context=$right_context ark:- ark:-|"
#  thread_string=
#  [ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 
#  mkdir -p $dir/decode/iter$iter/log
#  decode_dir=$dir/decode/iter$iter
#  $bin_dir/bd-nnet-forward --no-softmax=false --apply-log=true --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $mlp_test "$feats_test" ark:- 2>$decode_dir/log/forward.log | latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam \
#  --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$lang/words.txt \
#  $dir/final.mdl $lang/HCLG.fst ark:- "ark:|gzip -c > $decode_dir/lat.gz" 2> $decode_dir/log/decode.log || exit 1;
#  scripts/nnet1/score.sh $scoring_opts $data_test $lang $decode_dir || exit 1;
#  grep WER $decode_dir/wer_* | sort -k 2 | head -1 | awk -F ":" '{print $NF}'

  if [ $halving_count -eq 5 ]; then
      echo ""     
      echo "we support to stop training after adjust six times LR"
      echo ""
      break
  fi

  iter=$[$iter+1]
done

echo "$0 successfuly finished.. $dir"

sleep 3
exit 0
