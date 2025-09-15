import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def freeze_model(model, out_file): # freeze_graph.py
    #모델을 ConcreteFunction으로 변환
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    #동결함수 생성
    frozen_func = convert_variables_to_constants_v2(full_model)

    #동결 그래프(frozen graph) 저장
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./dnn",
                      name=out_file,
                      as_text=False)
