#!/bin/bash

dropout=(0.1 0.2 0.3 0.4 0.5)
data='./datasets/cifar10_whitened.t7'
#data='./datasets/cifar100_whitened.t7'

# resnet with different depth plus stochastic dropout
# res_depth=(11 47 164 227)
# for d in "${res_depth[@]}";do
# 	for k in "${dropout[@]}";do
# 		echo $d $k
# 		model=resnet-pre-act depth=$d stoDrop=$k nGPU=4 dataset=$data ./scripts/train_cifar.sh
# 	done
# done

# wide residual network with different widening factor with stochastic dropout
#width=(1 4 8 10)
#for w in "${width[@]}";do
#	for k in "${dropout[@]}";do
#		echo $w $k
#		model=wide-resnet widen_factor=$w depth=40 stoDrop=$k dataset=$data ./scripts/train_cifar.sh
#	done
#done

# wide residual network with different deepen factor
width=(1 2 4)
deepen=(2 3 4)
for w in "${width[@]}";do
	for d in "${deepen[@]}";do
		echo $d $w
		model=wide-resnet widen_factor=$w deepen_factor=$d depth=40 dataset=$data ./scripts/train_cifar.sh
	done
done

# wide residual network with different GPU number
#width=(1 4 8 10)
#for w in "${width[@]}";do
#	for k in "${dropout[@]}";do
#	echo $d $w $k
#	model=wide-resnet widen_factor=$w depth=16 stoDrop=$k dataset=$data ./scripts/train_cifar.sh
#done


