#!/bin/bash

# wide residual network with different widening factor
wrn_depth=(16 22 28)
for d in "${wrn_depth[@]}";do
	echo $d
	model=wide-resnet widen_factor=1 depth=$d ./scripts/train_cifar.sh
	for w in `seq 2 2 8`;do
		echo $w
		model=wide-resnet widen_factor=$w depth=$d ./scripts/train_cifar.sh
	done
done


# resnet with different depth
#res_depth=(11 47 164 227)
#for d in "${res_depth[@]}";do
#	echo $d
#	model=resnet-pre-act depth=$d ./scripts/train_cifar.sh
#done
