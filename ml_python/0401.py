#0401.py
# install : pip install tensorflow
'''
ref: 텐서플로 딥러닝 프로그래밍, 가메출판사
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

#1
x_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]], dtype = np.float32)
y_train = np.array([0,1,1,0], dtype = np.float32)   # XOR

y_train = tf.keras.utils.to_categorical(y_train) # one-hot

#2
model = tf.keras.Sequential([
    Input(shape=(2,)),
    Dense(units=4, activation='sigmoid', name='layer1'),
    Dense(units=2, activation='softmax', name = 'output')])
model.summary()

#3
#3-1
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

#3-2
ret = model.fit(x_train, y_train, epochs=100, verbose=2) # train, fit

#3-3: Checkpoint, # to load in  0402.py
##filepath = "./dnn/ckpt/0401-{epoch:04d}.ckpt"
##cp_callback = tf.keras.callbacks.ModelCheckpoint(
##              filepath, verbose=0, save_weights_only=False, save_freq=50)
##ret = model.fit(x_train, y_train, epochs=100, callbacks = [cp_callback], verbose=0)
##latest = tf.train.latest_checkpoint("./dnn/ckpt") # save_weights_only=True
##print('latest=', latest)

# 3-4: model save using Tensorflow SavedModel
model.save("./dnn/SAVED_MODEL") # to load in  0402.py, 0403.py

#4 모델 동결(freezing): pb 파일 생성, freeze_graph.py
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def freeze_model(model, out_file):
    #모델을 ConcreteFunction으로 변환
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    #동결함수 생성
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )
    frozen_func = convert_variables_to_constants_v2(full_model)

    #동결 그래프(frozen graph) 저장
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./dnn",
                      name=out_file,
                      as_text=False)
freeze_model(model, "XOR.pb")
