'''
Author: zhouyuchong
Date: 2024-04-16 13:20:44
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-04-16 13:53:17
'''
import os
import argparse
import tensorrt as trt


TRT_LOGGER = trt.Logger()



def build_engine(onnx_path, engine_type="fp16", dynamic_input=False):
    print("TensorRT version:", trt.__version__)
    version = int(trt.__version__.split(".")[0])
    engine_file_path = onnx_path.replace(".onnx", ".trt")

    if version == 10:
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
           0 
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28) # 256MiB
            # Parse model file
            if not os.path.exists(onnx_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_path))
            with open(onnx_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine
    else:
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
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
                profile.set_shape("input", (1,3,960,1280), (8,3,960,1280), (16,3,960,1280))
                config.add_optimization_profile(profile)

            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file, "wb") as f:
                f.write(serialized_engine)

        return True
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default="", required=True, help='onnx file path')
    parser.add_argument('--dynamic', action="stored_true", help='dynamic trt batch size')
    args = parser.parse_args()

    build_engine(args.onnx, "fp16", args.dynamic)

