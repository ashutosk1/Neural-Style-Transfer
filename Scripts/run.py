import os
import glob
import argparse
from model import NeuralStyleTransfer
from model import trainer
import utils


def main(config):    
    
    for idx in range(len(config["STYLE_IMAGE_PATH"])):
      style_image = utils.load_img(config["STYLE_IMAGE_PATH"][idx])
      config["CURR_STYLE"] = config["STYLES"][idx]  # Save curr style for future ref. 
      print(f"\n\t*** PERFORMING NEURAL STYLE TRANSFER FOR: {config['CURR_STYLE']} ***\t")


      # # Optimize the model on images/frames.
      # if config["IMAGE"]:
      #   assert len(config["STYLE_IMAGE_PATH"]) == len(config["STYLES"])
      #   content_image = utils.load_img(config["CONTENT_IMAGE_PATH"])  # Change it to Content Path
      #   model = NeuralStyleTransfer(config)
      #   trainer(model, content_image, style_image, config)
      
      # if config["VIDEO"]:
      #   pass
        # Extract Frames
      utils.extract_frames(config["CONTENT_IMAGE_PATH"], config["OUTPUT_TEMP"])
      # Get the Frames
      frame_files = sorted([f for f in os.listdir(config["OUTPUT_TEMP"]) if f.endswith('.jpg')])
      # Process frame-by-frame
      for i, frame in enumerate(frame_files):
          print(f"\t\tframe:{i+1}/{len(frame_files)}")
          content_image = utils.load_img(frame)
          model = NeuralStyleTransfer(config)
          trainer(model, content_image, style_image, config)
      
      utils.create_video_from_frames(config["OUTPUT_DIR"], 24, config)
      [os.remove(f) for f in glob.glob(os.path.join(config["OUTPUT_DIR"]), '*jpg')]




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Update image paths in config.json")
    parser.add_argument("--content_path", type=str, help="Path to the content image")
    parser.add_argument("--style_paths", type=str, nargs='+',help="Paths to style images (separate by spaces)")  # Allow multiple style paths
    parser.add_argument("--output_dir", type=str, default="../OUTPUT_IMAGES", help="Output Dir to save results")
    parser.add_argument("--save_image_at_epoch", type=utils.str_to_bool, default='N', help="Save image every epoch (Y/N, default: Y)")


    args =  parser.parse_args()
    config = utils.update_config(args)
    main(config)