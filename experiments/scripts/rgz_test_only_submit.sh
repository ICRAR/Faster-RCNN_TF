#!/bin/bash --login
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=pawsey0129
#SBATCH --constraint=tesla

#Default loaded compiler module is gcc module
#module load gcc
module load cuda
module unload gcc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/group/pawsey0129/software/sles11sp4/apps/gcc/4.8.5/cuda/7.5.18/lib64
export COMPILER=gcc
export COMPILER_VER=4.8.5
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGG16 rgz /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train/VGGnet_fast_rcnn-50000
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGG16 rgz /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_trainsecond/VGGnet_fast_rcnn-60000
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test rgz /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_trainthird/VGGnet_fast_rcnn-60000
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_testsmall rgzsmall /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_trainthirdsmall/VGGnet_fast_rcnn-60000
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test rgz /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_trainfourth/VGGnet_fast_rcnn-60000
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_testfifth rgzfifth /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_trainfifth/VGGnet_fast_rcnn-70000 --thresh 0.996
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_testsixth rgzsixth /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_trainsixth/VGGnet_fast_rcnn-140000 --thresh 0.965
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test07 rgz07 /home/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train07_secondround/VGGnet_fast_rcnn-60000  600 --force False
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test08 rgz08 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train08/VGGnet_fast_rcnn-130000 132 --thresh 0.5
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test09 rgz09 /home/cwu/rgz-ml-ws/code/output/faster_rcnn_end2end/rgz_2017_train09/VGGnet_fast_rcnn-50000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test11 rgz11 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train11/VGGnet_fast_rcnn-50000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test12 rgz12 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train12/VGGnet_fast_rcnn-55000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test13 rgz13 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train13/VGGnet_fast_rcnn-55000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test15 rgz15 /home/cwu/rgz-ml-ws/code/output/faster_rcnn_end2end/rgz_2017_train15/VGGnet_fast_rcnn-70000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test14 rgz14 /home/cwu/rgz-ml-ws/code/output/faster_rcnn_end2end/rgz_2017_train14/VGGnet_fast_rcnn-120000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test14 rgz14 /home/cwu/rgz-ml-ws/code/output/faster_rcnn_end2end/rgz_2017_train14/VGGnet_fast_rcnn-65000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test16 rgz16 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train16/VGGnet_fast_rcnn-70000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test14 rgz14 /home/cwu/rgz-ml-ws/code/output/faster_rcnn_end2end/rgz_2017_train14/VGGnet_fast_rcnn-15000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test17 rgz17 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train17/VGGnet_fast_rcnn-60000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test18 rgz18 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train18/VGGnet_fast_rcnn-65000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test08 rgz08 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train08/VGGnet_fast_rcnn-135000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test19 rgz19 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train19/VGGnet_fast_rcnn-125000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test16 rgz16  /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train16/VGGnet_fast_rcnn-160000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test14 rgz14 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train14_st_pool/VGGnet_fast_rcnn-115000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test22 rgz22 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train22/VGGnet_fast_rcnn-80000 600
/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test22 rgz22 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train22/VGGnet_fast_rcnn-70000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test22 rgz22 /home/cwu/rgz-ml-ws/code/output/faster_rcnn_end2end/rgz_2017_train22_gold/VGGnet_fast_rcnn-80000 600 --comp
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test21 rgz21 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train21/VGGnet_fast_rcnn-50000 600
#/group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_testing.sh gpu 0 VGGnet_test08 rgz08 /group/pawsey0129/cwu/rgz-faster-rcnn/output/faster_rcnn_end2end/rgz_2017_train08/VGGnet_fast_rcnn-150000 600
