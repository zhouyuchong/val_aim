'''
Author: zhouyuchong
Date: 2024-04-18 10:53:57
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-04-18 10:59:00
'''
import argparse
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

NETWORK_WIDTH = 640
NETWORK_HEIGHT = 640

def build_engine(onnx_path, engine_file, engine_type="fp16", dynamic_input=False):
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_batch_size = 16
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 27
        if engine_type.lower() == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return False
        if dynamic_input:
            profile = builder.create_optimization_profile()
            profile.set_shape("input", (1,3,NETWORK_WIDTH,NETWORK_HEIGHT), (1,3,NETWORK_WIDTH,NETWORK_HEIGHT), (1,3,NETWORK_WIDTH,NETWORK_HEIGHT))
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_file, "wb") as f:
            f.write(serialized_engine)

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default="", required=True, help='onnx file path')
    # parser.add_argument('--imgsz', type=str, default="", required=True, help='onnx file path')
    args = parser.parse_args()
    engine_file_path = args.onnx.replace(".onnx", ".engine")
    build_engine(args.onnx, engine_file_path, "fp16", True)
