import graphlab
import numpy as np

def get_numpy_data(data_sframe,features,output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return (features_matrix,output_array)

sales = graphlab.SFrame('D:\\ML_Learning\\UW_Regression\\Week2\\kc_house_data.gl\\')

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') # the [] around 'sqft_living' makes it a list
print example_features[0,:] # this accesses the first row of the data the ':' indicates 'all columns'
print example_output[0] # and the corresponding output

my_weights = np.ones(2)
my_features = example_features[0,:]
predicted_value = np.dot(my_weights,my_features)

def predict_output(feature_matrix,weights):
    pred = np.dot(feature_matrix,weights)
    return pred

test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0

def regression_GD(feature_matrix,output,initial_weights,step_size,tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        gradient_sum_square = 0
        for i in range(len(weights)):
            
        gradient_magnitude = sqrt(gradient_sum_square)
        if gradient_magnitude < tolerance:
            converged = True
    return weights

