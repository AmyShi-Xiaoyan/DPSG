## default installation from CRAN
install.packages("keras")

## As keras is just an interface to popular deep learning frameworks, we have to 
## install a specfic deep learning backend. The default and recommended backend 
## is TensorFlow. By calling install_keras(), it will install all the needed 
## dependencies for TensorFlow.
library(keras)
install_keras()

## Load MNIST dataset
mnist <- dataset_mnist()

str(mnist)

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

str(x_train)
str(y_train)

## Plot an image

index_image = 28 ## change this index to see different image.
input_matrix <- x_train[index_image,1:28,1:28]
output_matrix <- apply(input_matrix, 2, rev)
output_matrix <- t(output_matrix)
image(1:28, 1:28, output_matrix, col=gray.colors(256), xlab=paste('Image for digit of: ', y_train[index_image]), ylab="")


# step 1: reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# step 2: rescale
x_train <- x_train / 255
x_test <- x_test / 255
str(x_train)
str(x_test)

## Make y categorical variable

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
str(y_train)

## Define a neural network structure

dnn_model <- keras_model_sequential() 
dnn_model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(dnn_model)

## Compile the model;

dnn_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


## Train the neural network model;

dnn_history <- dnn_model %>% fit(
  x_train, y_train, 
  epochs = 15, batch_size = 128, 
  validation_split = 0.2
)

str(dnn_history)

plot(dnn_history)

## Prediction

dnn_model %>% evaluate(x_test, y_test)

dnn_pred <- dnn_model %>% 
              predict_classes(x_test)
head(dnn_pred, n=50)

## total number of mis-classcified images
sum(dnn_pred != mnist$test$y)

missed_image = mnist$test$x[dnn_pred != mnist$test$y,,]
missed_digit = mnist$test$y[dnn_pred != mnist$test$y]
missed_pred = dnn_pred[dnn_pred != mnist$test$y]

## Image for digit 5, wrongly predicted as 3;

index_image = 34 ## change this index to see different image.
input_matrix <- missed_image[index_image,1:28,1:28]
output_matrix <- apply(input_matrix, 2, rev)
output_matrix <- t(output_matrix)
image(1:28, 1:28, output_matrix, col=gray.colors(256), xlab=paste('Image for digit ', missed_digit[index_image], ', wrongly predicted as ', missed_pred[index_image]), ylab="")