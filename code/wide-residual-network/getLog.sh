#!/bin/bash

mkdir logFiles
for v in `ls logs`;do
	mkdir -p logFiles/$v
	cp /ogs/$v/log.txt logFiles/$v
done
