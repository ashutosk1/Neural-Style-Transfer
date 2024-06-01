import os
import glob
import argparse
from model import NeuralStyleTransfer
from model import trainer
import utils


def main(config):    
    
    assert len(config["STYLE_IMAGE_PATH"]) == len(config["STYLES"])

    for idx in range(len(config["STYLE_IMAGE_PATH"])):
      style_image = utils.load_img(config["STYLE_IMAGE_PATH"][idx])
      config["CURR_STYLE"] = config["STYLES"][idx]  # Save curr style for future ref. 
      print(f"\n\t*** PERFORMING NEURAL STYLE TRANSFER FOR: {config['CURR_STYLE']} ***\t")

      # Optimize the model on images
      if config["INPUT_TYPE"]==0:
        content_image = utils.load_img(config["CONTENT_PATH"])  # Change it to Content Path
        model = NeuralStyleTransfer(config)
        trainer(model, content_image, style_image, config)
      
      # Optimize the model for video-frames
      else:
        utils.extract_frames(config["CONTENT_PATH"], config["OUTPUT_TEMP_DIR"])
        # Get the Frames
        frame_files = sorted([os.path.abspath(os.path.join(config["OUTPUT_TEMP_DIR"], f)) \
                              for f in os.listdir(config["OUTPUT_TEMP_DIR"]) if f.endswith('.jpg')])
        #frame_files = frame_files[:10]  # Quick test

        # Style Transfer frame-by-frame
        for i, frame in enumerate(frame_files):
            print(f"\n\t\tframe: {i+1}/{len(frame_files)}")
            content_image = utils.load_img(frame)
            model = NeuralStyleTransfer(config)
            config["CURR_FRAME"] = i+1
            trainer(model, content_image, style_image, config)
        utils.create_video_from_frames(config["OUTPUT_DIR"], 24, config)
        [os.remove(f) for f in glob.glob(os.path.join(config["OUTPUT_DIR"], '*.jpg'))]
    
    if config["INPUT_TYPE"]==1:
      [os.remove(f) for f in glob.glob(os.path.join(config["OUTPUT_TEMP_DIR"], '*.jpg'))]


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Update params in config.json")
    parser.add_argument("--input_type", type=int, choices=[0, 1], default=0, help="Input type: 0 for image (default), 1 for video")
    parser.add_argument("--content_path", type=str, help="Path to the content image")
    parser.add_argument("--style_paths", type=str, nargs='+',help="Paths to style images (separate by spaces)")  # Allow multiple style paths
    parser.add_argument("--output_dir", type=str, default="../OUTPUT_DIR", help="Output Dir to save results")
    parser.add_argument("--save_image_at_epoch", type=utils.str_to_bool, default='N', help="Save image every epoch (Y/N, default: Y). \
                        Turn N when processing videos.")

    args =  parser.parse_args()
    config = utils.update_config(args)
    main(config)