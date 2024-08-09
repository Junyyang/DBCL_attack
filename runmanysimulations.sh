#!/bin/sh

echo "start running"
python3 train.py --datatype="LFW" --model_type="CNN"
python3 train.py --datatype="LFW"  --model_type="CNN_sketch"

echo "Hello world!"
