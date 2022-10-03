from flask import Flask, request, jsonify
import numpy as np
import random
from tensorflow import keras

def create_random_model_v2(input_shape, num_outputs):

  curr_model = keras.models.Sequential([
                                  keras.layers.Dense(6, input_shape=[input_shape]),   
                                  keras.layers.Dense(10, activation='relu'),
                                  keras.layers.Dense(10, activation='relu'),
                                  keras.layers.Dense(num_outputs, activation='sigmoid') #changed from 2
            ])
  
  model_weights = []
  for layer_index in range(len(curr_model.layers)):
    model_data = curr_model.layers[layer_index].get_weights()[0]

    for i in range(len(model_data)):
      for j in range(len(model_data[i])):
        model_data[i][j] = random.uniform(-1, 1)

    weight_framework = curr_model.layers[layer_index].get_weights()
    weight_framework[0] = model_data
    curr_model.layers[layer_index].set_weights(weight_framework)

  return curr_model

curr_model = create_random_model_v2(2, 1)

#importing a trained model:

layer_0 = np.array([[ 0.28652927, -0.327208  , -0.3626468 , -0.21121237,  0.35408235,
        -0.88698196],
       [ 0.7007187 ,  0.7769093 ,  0.7617503 ,  0.3843154 , -0.4712856 ,
        -0.9986683 ]])

layer_1 = np.array([[ 0.9849461 ,  0.93041724, -0.20599423, -0.90767235,  0.8673231 ,
        -0.0122947 ,  0.57239985, -0.809796  , -0.03562181,  0.45945457],
       [-0.9803096 , -0.72888047,  0.5235072 ,  0.10555113,  0.91488963,
        -0.06750356,  0.33303434,  0.5942374 , -0.40907326,  0.22085288],
       [ 0.85666   ,  0.5418823 ,  0.08330654,  0.8494281 ,  0.435585  ,
         0.5387422 ,  0.49194607, -0.27638105, -0.6776301 ,  0.951088  ],
       [-0.66276026, -0.39371297,  0.13389681,  0.17999029,  0.48920166,
         0.38535324,  0.26055914,  0.9858588 , -0.38818312, -0.8419503 ],
       [ 0.5130739 , -0.41034162,  0.06150073, -0.04132949,  0.25650674,
        -0.00847352, -0.9058767 ,  0.6200809 , -0.3769302 , -0.86461914],
       [-0.13249326,  0.3689914 , -0.36244723,  0.1163602 , -0.6930092 ,
        -0.6061633 ,  0.7333464 , -0.09681083,  0.19101383, -0.29942524]])

layer_2 = np.array([[ 0.11507208, -0.23028193, -0.9127067 ,  0.9961781 ,  0.44708022,
        -0.821738  ,  0.6295902 , -0.653332  , -0.31338423,  0.13265173],
       [-0.34605968,  0.52917176, -0.9222446 , -0.70390576, -0.9075368 ,
        -0.09011424, -0.23384741, -0.3701314 , -0.62783056, -0.6614834 ],
       [-0.22329473, -0.9049404 ,  0.89368683, -0.19280216,  0.584144  ,
         0.35961074, -0.73052335, -0.2524372 , -0.28794894, -0.01633653],
       [-0.7490422 , -0.20123127,  0.6929078 ,  0.32065576, -0.74086297,
        -0.15565734,  0.02877545, -0.75249183, -0.22156347,  0.7080693 ],
       [-0.8422688 , -0.06816848,  0.7123486 ,  0.94509923, -0.14916618,
        -0.32978052,  0.622594  ,  0.17390972, -0.17189579,  0.33425584],
       [ 0.2237808 , -0.5570095 , -0.4581545 , -0.32187232, -0.6133122 ,
        -0.13329946,  0.06085013, -0.18770967,  0.37426946, -0.8366167 ],
       [-0.63728786,  0.03442136, -0.30508888,  0.7336727 ,  0.88964564,
         0.22618875,  0.02748858,  0.26676124,  0.5514536 ,  0.7588546 ],
       [ 0.2650014 ,  0.23838903,  0.3287155 , -0.56154764, -0.8071689 ,
        -0.06605185,  0.8262892 , -0.91464   , -0.47678345, -0.41013536],
       [-0.59013224, -0.6893972 , -0.03411594, -0.54825693, -0.44464776,
        -0.95795715, -0.4776221 ,  0.85727817, -0.5257869 , -0.74137527],
       [ 0.17551728,  0.25195876,  0.31116003, -0.7229744 , -0.4553806 ,
        -0.5286123 ,  0.10832033,  0.93221587,  0.5712611 ,  0.17391807]],)


layer_3 = np.array([[-0.98576516],
       [-0.8871426 ],
       [ 0.3423495 ],
       [ 0.49945346],
       [-0.6698601 ],
       [-0.98061323],
       [ 0.36530834],
       [ 0.66649055],
       [ 0.319964  ],
       [ 0.8572223 ]])

layer_weights = [layer_0, layer_1, layer_2, layer_3]

for layer_index in range(len(layer_weights)):
  weight_framework = curr_model.layers[layer_index].get_weights()
  weight_framework[0] = layer_weights[layer_index]
  curr_model.layers[layer_index].set_weights(weight_framework)

print("MODEL SUCCSESSFULLY CREATED")

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def return_preds():
    data = request.get_json()
    output = curr_model.predict(np.asarray([data['position'], data['velocity']]).reshape(1, 2))
    return jsonify({'algo_output': str(output[0][0])})