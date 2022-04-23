
python static_quantize.py --input_model 0.05thr_pruned.onnx --output_model 0.05thr_pruned_stat_quant.onnx
python static_quantize.py --input_model 0.1thr_pruned.onnx --output_model 0.1thr_pruned_stat_quant.onnx
python static_quantize.py --input_model 0.2thr_pruned.onnx --output_model 0.2thr_pruned_stat_quant.onnx
python static_quantize.py --input_model 0.3thr_pruned.onnx --output_model 0.3thr_pruned_stat_quant.onnx
python static_quantize.py --input_model 0.4thr_pruned.onnx --output_model 0.4thr_pruned_stat_quant.onnx