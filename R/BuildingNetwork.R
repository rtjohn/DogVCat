# Building the network
library(keras)

model <- keras_model_sequential() %>% 
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                  input_shape = c(150, 150, 3)) %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    layer_flatten() %>% 
    layer_dense(units = 512, activation = 'relu') %>% 
    layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c('acc')
)
    
