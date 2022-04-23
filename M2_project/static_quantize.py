# python static_quantize.py --input_model 0.9thr_pruned.onnx --output_model 0.9thr_pruned_stat_quant.onnx

import os
import sys
import numpy as np
import re
import abc
import subprocess
import json
import argparse
import time
from PIL import Image

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, augmented_model_path='augmented_model.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            nhwc_data_list = preprocess_func(self.image_folder, height, width, size_limit=0)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))

        # Mean and standard deviation 
        mean = np.array((0.4914, 0.4822, 0.4465))
        std = np.array((0.2023, 0.1994, 0.2010))

        # normalize image
        input_data = (np.float32(pillow_img) / 255. - mean) / std

        # input_data = np.float32(pillow_img) - \
        # np.array([123.68, 116.78, 103.94], dtype=np.float32)

        nhwc_data = np.expand_dims(np.float32(input_data), axis=0)
        # nhwc_data = np.expand_dims(input_data, axis=0)

        nchw_data = nhwc_data.transpose([0, 3, 1, 2])
        # nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard

        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument("--calibrate_dataset", default="./test_deployment", help="calibration data set")
    parser.add_argument("--quant_format",
                        default=QuantFormat.QOperator,
                        type=QuantFormat.from_string,
                        choices=list(QuantFormat))
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    dr = ResNet50DataReader(calibration_dataset_path)
    quantize_static(input_model_path,
                    output_model_path,
                    dr,
                    quant_format=QuantFormat.QDQ,
                    per_channel=args.per_channel,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8)
    print('Calibrated and quantized model saved.')


if __name__ == '__main__':
    main()