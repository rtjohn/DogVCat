# Loading the VGG16 network with pretrained weights
model <- application_vgg16(weights = "imagenet")
# Note that you include the densely connected classifier on top; in all 
# previous cases, you discarded it.
img_path <- "~/DSwork/indian_elephant.jpg"

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
african_elephant_output <- model$output[, 387]