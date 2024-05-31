import os
import json
import argparse
from model import NeuralStyleTransfer
from model import trainer
import tensorflow as tf

import utils


def main(config):    
    assert len(config["STYLE_IMAGE_PATH"]) == len(config["STYLES"])
    content_image = utils.load_img(config["CONTENT_IMAGE_PATH"])

    for idx in range(len(config["STYLE_IMAGE_PATH"])):
      style_image = utils.load_img(config["STYLE_IMAGE_PATH"][idx])
      config["CURR_STYLE"] = config["STYLES"][idx]  # Save curr style for future ref. 
      print(f"\n\t*** PERFORMING NEURAL STYLE TRANSFER FOR: {config['CURR_STYLE']} ***\t")
      model = NeuralStyleTransfer(config)
      _ = trainer(model, content_image, style_image, config)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Update image paths in config.json")
    parser.add_argument("--content_path", type=str, help="Path to the content image")
    parser.add_argument("--style_paths", type=str, nargs='+',help="Paths to style images (separate by spaces)")  # Allow multiple style paths
    parser.add_argument("--output_dir", type=str, default="../OUTPUT_IMAGES", help="Output Dir to save results")
    parser.add_argument("--save_image_at_epoch", type=utils.str_to_bool, default='Y', help="Save image every epoch (Y/N, default: Y)")    
    args =  parser.parse_args()
    config = utils.update_config(args)
    main(config)