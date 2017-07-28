import graphlab
import numpy as np

#return 1+h0(xi)+h1(xi)+...., and output
def get_numpy_data(data_sframe,features,output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return (features_matrix,output_array)

#sales = graphlab.SFrame('D:\\ML_Learning\\UW_Regression\\Week2\\kc_house_data.gl\\')
sales = graphlab.SFrame('C:\\Machine_Learning\\kc_house_data.gl\\')


(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') # the [] around 'sqft_living' makes it a list
print example_features[0,:] # this accesses the first row of the data the ':' indicates 'all columns'
print example_output[0] # and the corresponding output

my_weights = np.ones(2)
my_features = example_features[0,:]
predicted_value = np.dot(my_weights,my_features)

#calculate the dot product of w*h
def predict_output(feature_matrix,weights):
    pred = np.dot(feature_matrix,weights)
    return pred

test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0

#2*(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)* [feature_i]
#2*error*[feature_i]
def feature_derivative(error,features):
    derivative = 2*np.dot(error,features)
    return derivative

my_weights = np.array([0., 0.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output
feature = example_features[:,0]
derivative = feature_derivative(errors,feature)
print derivative
print -np.sum(example_output)*2


from math import sqrt

def regression_GD(feature_matrix,output,initial_weights,step_size,tolerance):
    converged = False
    weights = np.array(initial_weights)
    count = 0;
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        prediction = predict_output(feature_matrix,weights)
        #print prediction
        # compute the errors as predictions - output
        errors = prediction - output
        #print errors
        gradient_sum_square = 0  # initialize the gradient sum of squares
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors,feature_matrix[:,i])
            #print derivative
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_square = gradient_sum_square + derivative * derivative
            #print gradient_sum_square
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size * derivative
            #print weights[i]
            # compute the square-root of the gradient sum of squares to get the gradient magnitude:            
            gradient_magnitude = sqrt(gradient_sum_square)
            #print gradient_magnitude
            count = count + 1
        #if gradient_magnitude < tolerance or count>10:
        if gradient_magnitude < tolerance:
            converged = True
    return weights

train_data,test_data = sales.random_split(.8,seed=0)
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7
weights_final = regression_GD(simple_feature_matrix,output,initial_weights,step_size,tolerance)
print weights_final

#run test using the first model
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
predict_value = predict_output(test_simple_feature_matrix,weights_final)
print predict_value


model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
weights_final = regression_GD(feature_matrix,output,initial_weights,step_size,tolerance)
print weights_final

#run test using the second model
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
predict_value_2 = predict_output(test_simple_feature_matrix,weights_final)
print predict_value_2

err1 = predict_value - test_output
err1 = err1*err1
rss1 = sqrt(err1.sum())
print rss1
 
err2 = predict_value_2 - test_output
err2 = err2*err2
rss2 = sqrt(err2.sum())
print rss2
