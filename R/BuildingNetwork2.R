model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", 
                  input_shape = c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4), metrics = c("acc")
)

dataGen <- image_data_generator(
    rescale = 1/255, 
    rotation_range = 40, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE
)

testDataGen <- image_data_generator(rescale = 1/255)

trainDir <- 'data/train'
valDir <- 'data/val'
# Validation data should not be augmented
trainGen <- flow_images_from_directory(
    trainDir,
    dataGen,
    target_size = c(150, 150),
    batch_size = 32,
    class_mode = "binary"
)

valGen <- flow_images_from_directory(
    valDir,
    testDataGen,
    target_size = c(150, 150),
    batch_size = 32,
    class_mode = "binary"
)

history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50
)


