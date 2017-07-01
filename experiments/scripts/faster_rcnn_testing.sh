#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=$1
DEV_ID=$2
NETWORK=$3 # e.g. VGGnet_test
DATASET=$4
NET_FINAL=$5
IMG_SIDE=$6

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

BASEDIR=/group/pawsey0129/cwu/rgz-faster-rcnn
PY_PATH=/group/pawsey0129/software/dlpyws/bin/python

case $IMG_SIDE in
  132)
    CONFIG_EXT="_132pix.yml"
    ;;
  600)
    CONFIG_EXT=".yml"
    ;;
  *)
    echo "No valid image side given"
    exit
    ;;
esac

case $DATASET in
  rgz)
    TRAIN_IMDB="rgz_2017_trainfourth"
    TEST_IMDB="rgz_2017_testfourth"
    PT_DIR="rgz"
    ITERS=60000
    ;;
  rgzsmall)
    TRAIN_IMDB="rgz_2017_trainthirdsmall"
    TEST_IMDB="rgz_2017_testthirdsmall"
    PT_DIR="rgz"
    ITERS=60000
    ;;
  rgzfifth)
    TRAIN_IMDB="rgz_2017_trainfifth"
    TEST_IMDB="rgz_2017_testfifth"
    PT_DIR="rgz"
    ITERS=70000
    ;;
  rgzsixth)
    TRAIN_IMDB="rgz_2017_trainsixth"
    TEST_IMDB="rgz_2017_testsixth"
    PT_DIR="rgz"
    ITERS=140000
    ;;
  rgz07)
    TRAIN_IMDB="rgz_2017_train07"
    TEST_IMDB="rgz_2017_test07"
    PT_DIR="rgz"
    ITERS=80000
    ;;
  rgz08)
    TRAIN_IMDB="rgz_2017_train08"
    TEST_IMDB="rgz_2017_test08"
    PT_DIR="rgz"
    ITERS=140000
    ;;
  rgz09)
    TRAIN_IMDB="rgz_2017_train09"
    TEST_IMDB="rgz_2017_test09"
    PT_DIR="rgz"
    ITERS=80000
    ;;
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# set +x
# NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
# set -x
# mycommand=$PY_PATH ${BASEDIR}/tools/test_net.py --device ${DEV} --device_id ${DEV_ID} \
#   --weights ${NET_FINAL} \
#   --imdb ${TEST_IMDB} \
#   --cfg ${BASEDIR}/experiments/cfgs/faster_rcnn_end2end.yml \
#   --network VGGnet_test \
#   ${EXTRA_ARGS}
#
# echo $mycommand

time $PY_PATH ${BASEDIR}/tools/test_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg ${BASEDIR}/experiments/cfgs/faster_rcnn_end2end${CONFIG_EXT} \
  --network ${NETWORK} \
  ${EXTRA_ARGS}
