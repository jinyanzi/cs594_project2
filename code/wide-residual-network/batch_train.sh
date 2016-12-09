#!/bin/bash

dropout=(0.1 0.2 0.3 0.4 0.5)
data='./datasets/cifar10_whitened.t7'
#data='./datasets/cifar100_whitened.t7'

# resnet with different depth
#res_depth=(11 47 164 227)
#for d in "${res_depth[@]}";do
#	echo $d
#	model=resnet-pre-act depth=$d dataset=$data ./scripts/train_cifar.sh
#done

# resnet with different depth plus stochastic dropout
#res_depth=(11 47 164 227)
#for d in "${res_depth[@]}";do
#	echo $d
#	model=resnet-pre-act depth=$d stoDrop=0.3 dataset=$data ./scripts/train_cifar.sh
#done

# wide residual network with different widening factor
#wrn_depth=(16 22 28 50)
#width=(1 2 4 8 10 12)
#for d in "${wrn_depth[@]}";do
#	for w in "${width[@]}";do
#		if [[ $d == 16 ]];then
#			if [[ $w == 1 || $w == 2 || $w == 4 || $w == 8 ]];then
#				continue
#			fi
#		fi
#		echo $d $w
#		model=wide-resnet widen_factor=$w depth=$d dataset=$data ./scripts/train_cifar.sh
#	done
#done

# wide residual network with different widening factor with regular dropout
wrn_depth=(16 22 40)
width=(1 4 8 10)
for d in "${wrn_depth[@]}";do
	for w in "${width[@]}";do
		for k in "${dropout[@]}";do
			echo $d $w $k
			model=wide-resnet widen_factor=$w depth=$d dropout=$k dataset=$data ./scripts/train_cifar.sh
		done
	done
done

# wide residual network with different widening factor with stochastic dropout
#wrn_depth=(16 22 28 50)
#width=(1 2 4 8 10 12)
#for d in "${wrn_depth[@]}";do
#	for w in "${width[@]}";do
#		for k in "${dropout[@]}";do
#		echo $d $w $k
#		model=wide-resnet widen_factor=$w depth=$d stoDrop=$k dataset=$data ./scripts/train_cifar.sh
#	done
#done

# compare speedup of number GPU
#nGPU=(1 2 3 4)
#for n in "${nGPU[@]}";do
#	echo $n
#	model=wide-resnet widen_factor=10 depth=52 dataset=./datasets/cifar100_whitened.t7 nGPU=$n./scripts/train_cifar.sh
#	model=resnet-pre-act depth=1001 dataset=./datasets/cifar100_whitened.t7 nGPU=$n ./scripts/train_cifar.sh
#done
