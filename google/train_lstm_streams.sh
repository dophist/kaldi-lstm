train_tool=bin/bd-nnet-train-lstm-streams

learn_rate=0.00001
momentum=0.9
num_stream=4
batch_size=20
targets_delay=5
dump_interval=10000

verbose=1
max_iters=15

start_halving=3
halving_factor=0.8

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 

. utils/parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: $0 <data-tr> <data-cv> <ali-train> <exp-dir>"
   echo " e.g.: $0 data/train data/train/tr data/train/cv exp/tri2a_ali exp/lstm"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

tr=$1
cv=$2
ali=$3
dir=$4

feats_tr="scp:$tr/feats.scp"
labels_tr="ark:gunzip -c $ali/ali.*.gz | ali-to-pdf $ali/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |"

feats_cv="scp:$cv/feats.scp"
labels_cv="ark:gunzip -c $ali/ali.*.gz | ali-to-pdf $ali/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |"

feature_transform=$dir/feature_transform.nnet.txt # for mean-var normalisation
nnet_init=$dir/nnet.init

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

if [ ! -f $nnet_init ]; then
    echo "Initializing lstm"
    nnet-initialize --binary=true $dir/nnet.proto $nnet_init
fi

cp $nnet_init $dir/nnet/nnet.iter0

iter=0
while [ $iter -lt $max_iters ]; do
    # learning rate decay
    if [ $iter -ge $start_halving ]; then
          learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    fi
    # train
    $train_tool \
        --feature-transform=$feature_transform \
        --learn-rate=$learn_rate \
        --momentum=$momentum \
        --num-stream=$num_stream \
        --batch-size=$batch_size \
        --targets-delay=$targets_delay \
        --dump-interval=$dump_interval \
        --verbose=$verbose \
        $feats_tr "$labels_tr" $dir/nnet/nnet.iter${iter} $dir/nnet/nnet.iter$[iter+1] \
        >& $dir/log/tr.iter$[iter+1].log

    # validate
    $train_tool \
        --cross-validate=true \
        --learn-rate=$learn_rate \
        --momentum=$momentum \
        --feature-transform=$feature_transform \
        --num-stream=$num_stream \
        --batch-size=$batch_size \
        --targets-delay=$targets_delay \
        --dump-interval=$dump_interval \
        --verbose=$verbose \
        $feats_cv "$labels_cv" $dir/nnet/nnet.iter$[iter+1] \
        >& $dir/log/cv.iter$[iter+1].log

    iter=$[iter+1]
done

echo "$0 successfuly finished.. $dir"
sleep 3
exit 0
