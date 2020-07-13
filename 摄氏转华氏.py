import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# 设置训练数据
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degree celsius is equals {} degree fahrenheit".format(c, fahrenheit_a[i]))


# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])]
)

# 使用损失和优化器函数编译模型
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam(0.1))

# 训练模型
history = model.fit(celsius_q, fahrenheit_a, epochs=5000, verbose=False)
print("train the model finished")

# 显示训练统计数据
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()

# 使用模型
print(model.predict([38]))

# 显示训练的模型参数值 [array([[1.7979493]], dtype=float32), array([31.952517], dtype=float32)]
print(model.get_weights())
