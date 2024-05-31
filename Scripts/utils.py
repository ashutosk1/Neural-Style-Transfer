import json
import os
import tensorflow as tf


def gram_matrix(input_tensor):
    """ 
    Calculates the Gram matrix of a feature map on the style outputs to get capture the style imformation.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def clip_0_1(image):
    """
    Ensures that the image pixel values are in [0, 1]
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def load_img(image_path):
    """
    Loads the Image from the given file-path and scaled such that
    the maximum dimension is max_dim.
    """
    max_dim = 512
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def update_config(args, config_file="config.json"):
    """
    Updates the config.json file with the provided Arguments. 
    Includes: (1). Content image path and a list of style image paths. 
                    Note: Paths can be single/multiple and abs/relative.
              (2). Output Dir
              (3). Flag for saving image at every iter(epoch).
              (4). Style Identifier List.
    """
    try:
      with open(config_file, 'r') as f:
        config = json.load(f)
    except FileNotFoundError:
      print("Config file not found.")
    

    # 1. Content Image Path
    content_path = os.path.abspath(args.content_path)
    style_paths   = [os.path.abspath(style_path) for style_path in args.style_paths]

    # 2. Output Dir
    output_dir = args.output_dir
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)  # Create Dir if it doesn't exist.

    # 3. Flag for saving Output Image
    save_image_at_epoch = args.save_image_at_epoch
    
    # 4. Styles Identifiers
    styles = [styles.split('/')[-1].split('.')[0] for styles in style_paths]

    # Update the `config.json` with updated args for future ref. 
    update_dicts = {             
                                 "CONTENT_IMAGE_PATH"   :  content_path,
                                  "STYLE_IMAGE_PATH"    :  style_paths,
                                  "OUTPUT_DIR"          :  output_dir,
                                  "SAVE_IMAGE_AT_EPOCH" :  save_image_at_epoch,
                                  "STYLES"              :  styles
                    }
    config.update(update_dicts)
    with open(config_file, 'w') as f:
      json.dump(config, f, indent=4)
    return config

    

def save_img(image, epoch, config):
    """ Save the Output Stylized Image at the end of every Epoch if the flag is True.
    """
    save_at_epoch = config["SAVE_IMAGE_AT_EPOCH"]
    is_final_epoch = (epoch == config["EPOCH"] - 1)
    
    if save_at_epoch or is_final_epoch:
        content_basename = config["CONTENT_IMAGE_PATH"].split('/')[-1].split('.')[0]
        style = config["CURR_STYLE"]
        filename = os.path.join(config["OUTPUT_DIR"], f"{content_basename}-{style}-{epoch:02d}.jpg")
        print(f"\tEPOCH: [{epoch+1}/{config['EPOCH']}]  SAVED RESULT IMAGE AT: {filename}.")
        tf.keras.preprocessing.image.save_img(filename, tf.squeeze(image).numpy())
    else:
        print(f"\tEPOCH: [{epoch+1}/{config['EPOCH']}]")


def str_to_bool(value):
    """Converts a string representation of truth to True or False.
    """
    if value.lower() in {'y', 'yes', 'true', 't', '1'}:
        return True
    elif value.lower() in {'n', 'no', 'false', 'f', '0'}:
        return False
    else:
        raise ValueError('Boolean value expected (Y/N).')