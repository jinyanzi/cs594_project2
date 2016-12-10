#!/bin/bash

model=wrn widen_factor=2 depth=16 stoDrop=0.3 ./scripts/train_cifar2.sh
#model=resnet-pre-act deepen_factor=3 depth=11 stoDrop=0.3 ./scripts/train_cifar2.sh
