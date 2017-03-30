#!/bin/bash
# Usage:
# ./experiments/scripts/colorchecker.sh GPU NET [options args to {train,test}_net.py]
# DATASET: Bayesian Color Constancy Revisited - Peter V. Gehler et al., CVPR 2008
#
# Example:
# ./experiments/scripts/colorchecker.sh 0 VGG_CNN_M_1024 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

# xtrace print commands
set -x
# errexit abort on first error
set -e

export PYTHONUNBUFFERED="True"

MY_NAME=${0%.sh}
MY_NAME=${MY_NAME##*\/}

GPU_ID=$1
NET=$2
# lower case
NET_lc=${NET,,}
DATASET="gehler"

# initialize an array
array=( $@ )
# number of elements in array ${#array_name[@]}
len=${#array[@]}
# more args after NET
EXTRA_ARGS=${array[@]:2:$len}
# replace space with underscore
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  gehler)
    TRAIN_IMDB="gehler_trainval"
    TEST_IMDB="gehler_test"
    PT_DIR="gehler"
    ITERS=60000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${MY_NAME}_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_FINAL="output/colorchecker/gehler_trainval/vgg16_colorchecker_iter_10000.caffemodel"
if [ -e $NET_FINAL ];then
    echo using $NET_FINAL
else
    time ./tools/train_net.py --gpu ${GPU_ID} \
      --solver models/${PT_DIR}/${NET}/${MY_NAME}/solver.prototxt \
      --weights data/imagenet_models/${NET}.v2.caffemodel \
      --imdb ${TRAIN_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${MY_NAME}.yml \
      ${EXTRA_ARGS}

    set +x
    NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
    set -x
fi

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/${MY_NAME}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/${MY_NAME}.yml \
  --vis \
  ${EXTRA_ARGS}
