from IPython.display import display, clear_output
import tensorflow as tf
import os
import tqdm
import utils

                 

class NeuralStyleTransfer(tf.keras.models.Model):
    """
    Neural Style Transfer model using pre-trained VGG19 for content and style feature extraction.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False


    def get_vgg_model_by_layers(self, layers):
        """
        Creates a sub-model of VGG19 based on specified layers. 
        Caution: Until the model is build or called with `self.call()`, `model.summary()` or `model.layers` will not 
                reflect only the desired layers.
        """
        outputs = [self.vgg.get_layer(name).output for name in layers]  # List of requisite output layers -> {Content Layers, Style Layers}. 
        vgg_model_by_layer =  tf.keras.Model([self.vgg.input], outputs)                  
        return vgg_model_by_layer    

    
    def call(self, inputs):
        """
        Forward Pass. 

        Steps:
        1. ** Preprocess Input **   -> Apply the VGG19 specific preprocessing on the scaled input in [1, 255.0].
        2. ** Extract Layers **     -> Extract the layers responsible for extracting features on content and styles.
        3. ** Get the Output **     -> Get the output for the extracted layers on the preprocessed Input. 
        4. ** Gram Matrixisation ** -> Apply Gram-matrix function on the style outputs to get capture the style imformation.
        5. ** Output **             -> Returns a dictionary of dictionary with (content_layers : content_output) and (style_layers : style_outputs).  
        """
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs*255.0)

        vgg_model_by_layer = self.get_vgg_model_by_layers(self.config["STYLE_LAYERS"] +
                                                          self.config["CONTENT_LAYERS"])
        outputs = vgg_model_by_layer(preprocessed_input)
        style_outputs = outputs[:len(self.config["STYLE_LAYERS"])]
        content_outputs = outputs[len(self.config["STYLE_LAYERS"]):]
        style_outputs = [utils.gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {
                        content_name: value for \
                        content_name, value in \
                        zip(self.config["CONTENT_LAYERS"], content_outputs)
                    }
        
        style_dict  = {
                        style_name: value for \
                        style_name, value in \
                        zip(self.config["STYLE_LAYERS"], style_outputs)
                    }
        
        return      {
                    'content': content_dict, 
                    'style': style_dict
                     }
    

def loss_fn(content_outputs, content_targets, style_outputs, style_targets, config):
    """
    Calculates the combined loss function for style transfer. 
    """
    # print(f"CONTENET TARGET: {content_targets}")
    # print(f"CONTENET OUTPUT: {content_outputs}")

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= config["STYLE_WEIGHT"] / len(config["STYLE_LAYERS"])  

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= config["CONTENT_WEIGHT"] / len(config["CONTENT_LAYERS"]) 

    loss = style_loss + content_loss
    return loss

        
# @tf.function()
# def train_step(model, optimizer, stylized_image, content_targets, style_targets, config):
#     """
#     Performs a single training step for the Neural Style Transfer model.
#     This function executes a single training step by:
#     1. Calculating the loss using the loss_fn function.
#     2. Applying gradients to the image using the optimizer.
#     3. Clipping the image values between 0 and 1.
#     """
#     with tf.GradientTape() as tape:
#         outputs = model(stylized_image)
#         # Get the Content and Style components of the outputs of the `model.call()` on tf.Variable.
#         content_outputs = outputs['content']
#         style_outputs = outputs['style']
#         loss = loss_fn(content_outputs, content_targets, style_outputs, style_targets, config)
#     grad = tape.gradient(loss, stylized_image)
#     optimizer.apply_gradients([(grad, stylized_image)])
#     stylized_image.assign(utils.clip_0_1(stylized_image))
    

# def trainer(model, content_image, style_image, config):
#     """
#     Performs the training loop for the Neural Style Transfer model. 
#     """
#     # Variable Image
#     stylized_image = tf.Variable(content_image)

#     # Static Targets
#     content_targets = model(content_image)['content']
#     style_targets = model(style_image)['style']
  
#     #Optimizer
#     optimizer = tf.keras.optimizers.Adam(config["LR"], config["BETA_1"], config["EPSILON"])

#     print(f"OPT:{optimizer}")

#     step = 0
#     for n in range(config["EPOCH"]):
#         for m in range(config["STEPS_PER_EPOCH"]):
#             step += 1
#             train_step(model, optimizer, stylized_image, content_targets, style_targets, config)
#             print(".", end='', flush=True)


class TrainStep(tf.Module):
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    @tf.function
    def __call__(self, stylized_image, content_targets, style_targets):
        with tf.GradientTape() as tape:
            outputs = self.model(stylized_image)
            content_outputs = outputs['content']
            style_outputs = outputs['style']
            loss = loss_fn(content_outputs, content_targets, style_outputs, style_targets, self.config)
        grad = tape.gradient(loss, stylized_image)
        self.optimizer.apply_gradients([(grad, stylized_image)])
        stylized_image.assign(utils.clip_0_1(stylized_image))
        return stylized_image
    
    
def trainer(model, content_image, style_image, config, style):
    stylized_image = tf.Variable(content_image)
    content_targets = model(content_image)['content']
    style_targets = model(style_image)['style']

    optimizer = tf.keras.optimizers.Adam(config["LR"], config["BETA_1"], config["EPSILON"])
    train_step = TrainStep(model, optimizer, config)

    step = 0
    for n in (range(config["EPOCH"])):
        for m in range(config["STEPS_PER_EPOCH"]):
            step += 1
            train_step(stylized_image, content_targets, style_targets)
            print(".", end='', flush=True)

    output_dir = config["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    content_basename = os.path.basename(config["CONTENT_IMAGE_PATH"])
    filename = os.path.join(config["OUTPUT_DIR"], f"{content_basename}-{style}.jpg")
    tf.keras.preprocessing.image.save_img(filename, tf.squeeze(stylized_image).numpy())
    print(f"SAVED RESULT IMAGE at: {filename}")