#!/bin/bash

python convert_to_onnx.py --model="0.05thr_pruned"
python convert_to_onnx.py --model="0.1thr_pruned"
python convert_to_onnx.py --model="0.2thr_pruned"
python convert_to_onnx.py --model="0.3thr_pruned"
python convert_to_onnx.py --model="0.4thr_pruned"
python convert_to_onnx.py --model="0.5thr_pruned"
python convert_to_onnx.py --model="0.6thr_pruned"
python convert_to_onnx.py --model="0.7thr_pruned"
python convert_to_onnx.py --model="0.8thr_pruned"
python convert_to_onnx.py --model="0.9thr_pruned"
