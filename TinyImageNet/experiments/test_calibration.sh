#!/bin/bash

source ~/new_begin.sh

python test_calibration.py --method_name calib_$1 --num_to_avg 10 > snapshots/calib_$1_test.txt
# allconv_calib_oe_scratch, wrn_calib_baseline, wrn_calib_oe_tune

