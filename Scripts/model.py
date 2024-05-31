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
    Calculates the content and style loss function for style transfer. 
    """
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= config["STYLE_WEIGHT"] / len(config["STYLE_LAYERS"])  

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= config["CONTENT_WEIGHT"] / len(config["CONTENT_LAYERS"]) 
    
    loss = style_loss + content_loss
    return loss


def variation_loss(stylized_image, config):
    """
    calculates the variation loss which will feed into `loss_fn' to get the combined loss
    """
    variation_loss = config["VARIATION_WEIGHT"]*tf.image.total_variation(stylized_image)
    return variation_loss


class TrainStep(tf.Module):
    """
    Performs a single training step for neural style transfer.
    Note: Creating a seperate class of TrinStep avoids the following ValueError:
            tf.function only supports singleton tf.Variables created once on the first call, 
            and reused across subsequent function calls.
        This was important so as to process the same content image with multiple styles, which
        encountered this error because of the Optimizer - which in the backend creates a tf.Variables
        and conflicts with re-processing of the image in the next iteration with a different style.
    """
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
            style_and_content_loss = loss_fn(content_outputs, content_targets, style_outputs, style_targets, self.config)
            var_loss = variation_loss(stylized_image, self.config)
            loss = style_and_content_loss + var_loss
        grad = tape.gradient(loss, stylized_image)
        self.optimizer.apply_gradients([(grad, stylized_image)])
        stylized_image.assign(utils.clip_0_1(stylized_image))
        return stylized_image
    
    
def trainer(model, content_image, style_image, config):
    """
    Trains the model to apply style from style_image to content_image.
    """
    stylized_image = tf.Variable(content_image)
    content_targets = model(content_image)['content']
    style_targets = model(style_image)['style']

    optimizer = tf.keras.optimizers.Adam(config["LR"], config["BETA_1"], config["EPSILON"])
    train_step = TrainStep(model, optimizer, config)

    output_dir = config["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    step = 0
    for n in (range(config["EPOCH"])):
        for _ in range(config["STEPS_PER_EPOCH"]):
            step += 1
            train_step(stylized_image, content_targets, style_targets)
            print(".", end='', flush=True)
        utils.save_img(stylized_image, n, config)