#!/bin/bash

#model=wide-resnet widen_factor=2 deepen_factor=3 depth=13 stoDrop=0.3 ./scripts/train_cifar2.sh
model=resnet-pre-act depth=11 stoDrop=0.3 ./scripts/train_cifar2.sh
