#!/bin/bash

python convert_to_onnx.py --model="5percent_pruned_zeroelim"
python convert_to_onnx.py --model="10percent_pruned_zeroelim"
python convert_to_onnx.py --model="20percent_pruned_zeroelim"
python convert_to_onnx.py --model="30percent_pruned_zeroelim"
python convert_to_onnx.py --model="40percent_pruned_zeroelim"
python convert_to_onnx.py --model="50percent_pruned_zeroelim"
python convert_to_onnx.py --model="60percent_pruned_zeroelim"
python convert_to_onnx.py --model="70percent_pruned_zeroelim"
python convert_to_onnx.py --model="80percent_pruned_zeroelim"
# python convert_to_onnx.py --model="90percent_pruned_zeroelim"
