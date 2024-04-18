<!--
 * @Author: zhouyuchong
 * @Date: 2024-04-16 13:16:56
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-04-18 16:32:54
-->
# val_aim

use YOLOV9 for valorant detection

## Usage
```
python3 script/trt_test.py --trt model/yolov9_val.engine --img data/images --conf 0.1 --topk 2
```

## Output
<img src="/output/v00_trt_out.jpg" width="600">

## TODO
+ use obs to push stream and infer
+ find a way to control mouse(not win32 api)

## Reference

[TensorRT Script](https://github.com/NVIDIA/TensorRT/tree/release/10.0/samples/python)

[YoloV9](https://github.com/WongKinYiu/yolov9)