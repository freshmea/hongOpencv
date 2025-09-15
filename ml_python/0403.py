#0403.py
'''
pip install onnx
pip install tf2onnx
pip install onnxruntime

'''
import numpy as np
import tensorflow as tf

#1: load SavedModel
model = tf.keras.models.load_model("./dnn/SAVED_MODEL.keras") # saved in 0401.py
model.summary()

#2
X = np.array([[0, 0], [0, 1], [1, 0],  [1, 1]], dtype = np.float32)

outs = model(X)
pred = np.argmax(outs, axis = 1)
print('pred=', pred)

import onnx

model.output_names = [t.name.split(":")[0] for t in model.outputs]
#3:  model -> ONNX ( pip install tf2onnx )
#3-1
import tf2onnx

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, "./dnn/XOR_tf.onnx")

#3-2
spec = (tf.TensorSpec((None, 2), tf.float32, name='input'),)
onnx_model, _ = tf2onnx.convert.from_keras(model,
                                       input_signature=spec,
                                       output_path="./dnn/XOR_tf2.onnx")
output_names = [n.name for n in onnx_model.graph.output]

#3-4: check model
onnx_model=onnx.load("./dnn/XOR_tf2.onnx")
onnx.checker.check_model(onnx_model)

#3-3: cmd
#python -m tf2onnx.convert --saved-model  ./SAVED_MODEL --output  XOR_tf3.onnx

#4: pip install onnxruntime
#ref:https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import onnxruntime as ort

ort_sess = ort.InferenceSession("./dnn/XOR_tf.onnx")

ort_inputs = {ort_sess.get_inputs()[0].name: X}
ort_outs = ort_sess.run(None, ort_inputs)[0]
##ort_outs = ort_sess.run(output_names, {'input': X})[0]

ort_pred = np.argmax(ort_outs, axis = 1)
print('ort_pred=', ort_pred)
