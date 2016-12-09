#!/bin/bash

model=wrn widen_factor=1 depth=16 stoDrop=0.3 ./scripts/train_cifar2.sh
