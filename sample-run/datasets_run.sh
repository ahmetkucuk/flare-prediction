#!/usr/bin/env bash
for VARIABLE in "12_6" "12_12" "12_24" "24_6" "24_12" "24_24"
do
	export DATASETNAME=$VARIABLE
	nohup ./run.sh > out_$VARIABLE 2>&1 &
done