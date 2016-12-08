#!/bin/bash

# wide residual network with different widening factor
wrn_depth=(22 28)
width=(1 2 4 8 10 12)
for d in "${wrn_depth[@]}";do
	for w in "${width[@]}";do
		if [[ $d == 22 ]];then
			if [[ $w == 1 || $w == 2 ]];then
				continue
			fi
		fi
		echo $d $w
		model=wide-resnet widen_factor=$w depth=$d ./scripts/train_cifar.sh
	done
done


# resnet with different depth
#res_depth=(11 47 164 227)
#for d in "${res_depth[@]}";do
#	echo $d
#	model=resnet-pre-act depth=$d ./scripts/train_cifar.sh
#done
