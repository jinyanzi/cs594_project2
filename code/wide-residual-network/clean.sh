#!/bin/bash

for i in `seq 1 9`;
	do echo $i;
	for v in `ls -d logs/wide-resnet_$i*`;do
		echo $v
		rm -rf $v;
	done
done
