# Building the network
# TO DO
    # Specify my training and validation directories as trainDir, etc.
    # Shrink the data size for a quick test
#library(keras)
#library(cloudml)
#library(beepr)
#start_time <- Sys.time()
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

# Stream in the data
trainDataGen <- image_data_generator(rescale = 1/255)
valDataGen <- image_data_generator(rescale = 1/255)
# Specify the location of my directories
trainDir <- 'data/train'
valDir <- 'data/val'
# Construct the generators
# QUESTION - there are two directories in the trainDir.
# Does the generator randomly pull from each directory in there?
# ANSWER -  yes!  It does.  Per the man page:
# path to the target directory. It should contain one subdirectory per class.
trainGenerator <- flow_images_from_directory(
    trainDir,
    trainDataGen,
    target_size = c(150,150),
    batch_size = 20,
    class_mode = "binary"
)
valGenerator <- flow_images_from_directory(
    valDir,
    valDataGen,
    target_size = c(150,150),
    batch_size = 20,
    class_mode = "binary"
)
history <- model %>% fit_generator(
    trainGenerator,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = valGenerator,
    validation_steps = 50
)
# beep()
# end_time <- Sys.time()
# end_time - start_time
# Time difference of 45.49059 mins
# model %>% save_model_hdf5("models/cvd_1.h5")

