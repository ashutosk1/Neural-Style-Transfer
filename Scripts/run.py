import os
import json
import argparse
from model import NeuralStyleTransfer
from model import trainer
import tensorflow as tf
import gc

import utils

def update_config(content_path:str, style_paths:str, config_file="config.json"):
    """
    Updates the config.json file with the provided content image path and a list of style image paths.
    Note: The paths to style images can be relative. Therefore, we can apply different styles on the same content image.
    """
    content_path = os.path.abspath(content_path)
    style_paths   = [os.path.abspath(style_path) for style_path in style_paths]
    
    try:
      with open(config_file, 'r') as f:
        config = json.load(f)
    except FileNotFoundError:
      print("Config file not found.")

    # Create `OUTPUT_IMAGE_DIR` if it doesn't exist.
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    
    # Update the `config.json` with content and style image paths for future ref. 
    add_path_and_style_dicts = {
                                  "CONTENT_IMAGE_PATH" : content_path,
                                  "STYLE_IMAGE_PATH"  :  style_paths
                                }
    config.update(add_path_and_style_dicts)
    # Save the updated config
    with open(config_file, 'w') as f:
      json.dump(config, f, indent=4)
    
    return config


def main(config):
    
    content_image = utils.load_img(config["CONTENT_IMAGE_PATH"])
    for idx in range(len(config["STYLE_IMAGE_PATH"])):
      style_image_path = config["STYLE_IMAGE_PATH"][idx]
      style_image = utils.load_img(style_image_path)
      style = style_image_path.split('/')[-1].split('.')[0]
      print(f"STYLE: {style}")
      model = NeuralStyleTransfer(config)
      _ = trainer(model, content_image, style_image, config, style)


if __name__=="__main__":
    with open('config.json', 'r') as f:
        config= json.load(f)
    
    parser = argparse.ArgumentParser(description="Update image paths in config.json")
    parser.add_argument("content_path", type=str, help="Path to the content image")
    parser.add_argument("style_paths", type=str, nargs="+", help="Paths to style images (separate by spaces)")  # Allow multiple style paths
    args = parser.parse_args()

    config = update_config(args.content_path, args.style_paths)
    main(config)