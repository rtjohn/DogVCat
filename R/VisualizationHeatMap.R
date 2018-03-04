library(keras)
# Loading the VGG16 network with pretrained weights
model <- application_vgg16(weights = "imagenet")
# Note that you include the densely connected classifier on top; in all 
# previous cases, you discarded it.
img_path <- "~/DSwork/Cat03.jpg"

# Process the image
img <- image_load(img_path, target_size = c(224, 224)) %>% # Convert to 224×224
    image_to_array() %>% # Array of shape (224, 224, 3)
    # Adds a dimension to transform the array
    # into a batch of size (1, 224, 224, 3)
    array_reshape(dim = c(1, 224, 224, 3)) %>%
    # Preprocesses the batch (this does channel-wise color normalization)
    imagenet_preprocess_input()
# Run the pretrained network on the image and decode its prediction vector 
# back to a human-readable format:
preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]
which.max(preds[1,])
# Setting up GRAD-Cam algorithm
# "Tusker" entry in prediction vector
tusker_output <- model$output[, 283]
# Output feature map of the block5_conv3 layer, 
# the last convolutional layer in VGG16
last_conv_layer <- model %>% get_layer("block5_conv3")
# Gradient of the “African elephant” class with regard to the output 
# feature map of block5_conv3
grads <- k_gradients(tusker_output, last_conv_layer$output)[[1]]
# Vector of shape (512) where each entry is the mean intensity of 
# the gradient over a specific feature-map channel
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
# Lets you access the values of the quantities you just defined: 
# pooled_grads and the output feature-map of block5_conv3, 
# given a sample image
iterate <- k_function(list(model$input),
                      list(pooled_grads, last_conv_layer$output[1,,,]))
# Values of these two quantities, given the sample image 
# of an Indian elephant
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
# Multiplies each channel in the feature-map array by 
# “how important this channel is” with regard to the “tusker” class
for (i in 1:512) {
    conv_layer_output_value[,,i] <-
        conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}
# The channel-wise mean of the resulting feature-map is the heatmap of the 
# class activation.
heatmap <- apply(conv_layer_output_value, c(1,2), mean)
# For visualization purposes, you’ll also normalize the heatmap between 0 and 1. 
# Heatmap Post-processing
heatmap <- pmax(heatmap, 0)
# Normalizes between 0 and 1
heatmap <- heatmap / max(heatmap)
# Function to write a heatmap to a PNG
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
    png(filename, width = width, height = height, bg = bg)
    op = par(mar = c(0,0,0,0))
    on.exit({par(op); dev.off()}, add = TRUE)
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}
# Writes the heatmap
write_heatmap(heatmap, "cat_heatmap.png")

# Superimposing the heatmap with the original picture
library(magick)
library(viridis)
# Reads the original elephant image and its geometry
image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)
# Creates a blended/ transparent version of the heatmap image
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "cat_overlay.png",
              width = 14, height = 14, bg = NA, col = pal_col)
# Overlays the heatmap
image_read("cat_overlay.png") %>%
    image_resize(geometry, filter = "quadratic") %>%
    image_composite(image, operator = "blend", compose_args = "20") %>%
    plot()





