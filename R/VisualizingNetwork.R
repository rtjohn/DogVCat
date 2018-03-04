# Visualizing what the convnet is doing?----------------------------
library(keras)
model <- load_model_hdf5("models/cvd_2.h5")
model
# Processing a single image---------------------------------------
imgPath <- "~/DSWork/dvcSmall/test/cats/cat.1700.jpg"
img <- image_load(imgPath, target_size = c(150,150))
imgTensor <- image_to_array(img)
imgTensor <- array_reshape(imgTensor, c(1, 150, 150, 3))
imgTensor <- imgTensor/255
dim(imgTensor)
# Display the test picture
plot(as.raster(imgTensor[1,,,]))

# Instantiating a model from an input tensor and a list of output tensors
# Extracts the outputs of the top 8 layers
layerOutputs <- lapply(model$layers[1:8], function(layer) layer$output)
# Creates a model that will return these outputs, given an input
activationModel <- keras_model(inputs = model$input, outputs = layerOutputs)
# Returns a list fo 5 arrays, one per layer activation
activations <- activationModel %>% predict(imgTensor)
# Activations of the first conv layer
firstLayerActivation <- activations[[1]]
dim(firstLayerActivation)
# 148x148 feature map with 32 channels
# Create a function to plot a channel
plotChannel <- function(channel){
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(channel), axes = FALSE, asp = 1,
          col = terrain.colors(8))
}
# Plotting the second channel
plotChannel(firstLayerActivation[1,,,18])

# Visualizing every channel in every intermediate activation
imageSize <- 58
imagesPerRow <- 16

for (i in 1:8) {
    layerActivation <- activations[[i]]
    layerName <- model$layers[[i]]$name
    
    nFeatures <- dim(layerActivation)[[4]]
    nCols <- nFeatures %/% imagesPerRow
    
    png(paste0("catActivations ", i, "_", layerName, ".png"),
        width = imageSize * imagesPerRow,
        height = imageSize * nCols)
    op <- par(mfrow = c(nCols, imagesPerRow), mai = rep_len(0.02, 4))
    
    for (col in 0:(nCols - 1)) {
        for (row in 0:(imagesPerRow - 1)) {
            channelImage <- layerActivation[1,,,(col*imagesPerRow) + row + 1]
            plotChannel(channelImage)
        }
    }
    
    par(op)
    dev.off()
}

# Visualizing convnet filters
# Defining the loss tensor for filter visualization
model <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE
)

layerName <- "block3_conv1"
filterIndex <- 1

layerOutput <- get_layer(model, layerName)$output
loss <- k_mean(layerOutput[,,,filterIndex])
# Obtaining the gradient of the loss with regard to the input
# Call to k_gradients returns an R list of tensors (of size 1 in this case).
# Hence you keep only the first element, which is a tensor
grads <- k_gradients(loss, model$input)[[1]]
# Gradient-normalization trick
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
# Fetching output values given input values
iterate <- k_function(list(model$input), list(loss, grads))

c(lossValue, gradsValue) %<-%
    iterate(list(array(0, dim = c(1, 150, 150, 3))))
# Loss maximization via stochastic gradient descent
# Starts from a gray image with some noise
inputImgData <- 
    array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128
# Runs gradient ascent for 40 steps
step <- 1
for (i in 1:40) {
    # Computes the loss value and gradient value
    c(lossValue, gradsValue) %<-% iterate(list(inputImgData))
    # Adjust the input image in the direction that maximizes loss
    inputImgData <- inputImgData = (gradsValue * step)
}
# Utility function to convert a tensor into a valid image
deprocess_image <- function(x) {
    dms <- dim(x)
    x <- x - mean(x)
    x <- x / (sd(x) + 1e-5)
    x <- x * 0.1
    x <- x + 0.5
    x <- pmax(0, pmin(x, 1))
    array(x, dim = dms)
}
# Function to generate filter visualizations
generate_pattern <- function(layer_name, filter_index, size = 150) {
    # Builds a loss function that maximizes the activation of the nth filter 
    # of the layer under consideration
    layer_output <- model$get_layer(layer_name)$output
    loss <- k_mean(layer_output[,,,filter_index])
    # Computes the gradient of the input picture with regard to this loss
    grads <- k_gradients(loss, model$input)[[1]]
    # Normalization trick: normalizes the gradient
    grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
    # Returns the loss and grads given the input picture
    iterate <- k_function(list(model$input), list(loss, grads))
    # Starts from a gray image with some noise
    input_img_data <-
        array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
    # Runs gradient ascent for 40 steps
    step <- 1
    for (i in 1:40) {
        c(loss_value, grads_value) %<-% iterate(list(input_img_data))
        input_img_data <- input_img_data + (grads_value * step)
    }
    img <- input_img_data[1,,,]
    deprocess_image(img)
}
library(grid)
grid.raster(generate_pattern("block3_conv1", 1))
# Generating a grid of all filter response patterns in a layer
library(grid)
library(gridExtra)
dir.create("vgg_filters")
for (layer_name in c("block1_conv1", "block2_conv1",
                     "block3_conv1", "block4_conv1")) {
    size <- 140
    png(paste0("vgg_filters/", layer_name, ".png"),
        width = 8 * size, height = 8 * size)
    grobs <- list()
    for (i in 0:7) {
        for (j in 0:7) {
            pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
            grob <- rasterGrob(pattern,width = unit(0.9, "npc"),
                               height = unit(0.9, "npc"))
            grobs[[length(grobs)+1]] <- grob
        } }
    grid.arrange(grobs = grobs, ncol = 8)
    dev.off()
}
    
    
    
    

