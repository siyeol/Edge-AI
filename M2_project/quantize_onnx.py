import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse

parser = argparse.ArgumentParser(description='M1')

parser.add_argument('--model', type=str, default="0.05thr_pruned", help='Model to convert')
args = parser.parse_args()


model_fp32 = args.model + ".onnx"
model_quant = args.model + "_quant" + ".onnx"

quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)