
# Neural-Style Transfer

This project is an attempt to implement Neural Style Transfer using TensorFlow with an objective to transform the content image in the artistic style of another image. 


This uses an optimization technique where content (input image) and style (reference image) losses are calculated and minimized to retain the content characterstics of the input image but blend it with the artistic style of the reference image.



## Features

- Pre-trained VGG19: Leverages a pre-trained VGG19 model for efficient feature extraction.
- User-Defined Style Selection: Flexibility to select any image as the style reference.
- Multi-style Experimentation: Facilitates the application of multiple artistic styles, one at a time, to a single content image. 
- CLI-based Experimentation: Simplifying the configuration and execution process. 



## Usage/Examples

1. Clone the repository
```
https://github.com/ashutosk1/Neural-Style-Transfer.git
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. The project utilizes a CLI tool for configuration and execution.

Available Options:
* `content_path` : Path to the content image
* `style_paths`  : Paths to style images (separate by spaces). Supports single-multiple style images with abs/relative paths. 
* `output_dir`   : Output Dir to save results.
* `save_image_at_epoch`: Save image every epoch (Y/N, default: Y).

Sample Command
```
cd ./Scripts
python3 run.py --content_path=../Examples/img.jpg \
--style_paths ../Examples/Styles/picasso-style.jpg \
../Examples/Styles/van-gogh-style.jpg \
../Examples/Styles/edward-munch-style.jpg \
--output_dir ../out --save_image_at_epoch Y
```




## Results

<table>
  <tr>
    <td style="vertical-align: top;">
      <img src="https://raw.githubusercontent.com/ashutosk1/Neural-Style-Transfer/main/Examples/Styles/edward-munch-style.jpg" alt="Edward-Munch" height="300"/>
      <br>
      <sub>Edward-Munch's The Scream</sub>
    </td>
    <td style="vertical-align: top;">
      <img src="https://raw.githubusercontent.com/ashutosk1/Neural-Style-Transfer/main/Examples/output_edward.gif" alt="Style Transfer Progress" height="300"/>
      <br>
      <sub>Style Transfer Progress</sub>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td style="vertical-align: top;">
      <img src="https://raw.githubusercontent.com/ashutosk1/Neural-Style-Transfer/main/Examples/Styles/picasso-style.jpg" alt="Edward-Munch" height="300"/>
      <br>
      <sub>Picasso's Self-Potrait</sub>
    </td>
    <td style="vertical-align: top;">
      <img src="https://raw.githubusercontent.com/ashutosk1/Neural-Style-Transfer/main/Examples/output_picasso.gif" alt="Style Transfer Progress" height="300"/>
      <br>
      <sub>Style Transfer Progress</sub>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td style="vertical-align: top;">
      <img src="https://raw.githubusercontent.com/ashutosk1/Neural-Style-Transfer/main/Examples/Styles/van-gogh-style.jpg" alt="Edward-Munch" height="300"/>
      <br>
      <sub>Van Gogh's Starry Night</sub>
    </td>
    <td style="vertical-align: top;">
      <img src="https://raw.githubusercontent.com/ashutosk1/Neural-Style-Transfer/main/Examples/output_gogh.gif" alt="Style Transfer Progress" height="300"/>
      <br>
      <sub>Style Transfer Progress</sub>
    </td>
  </tr>
</table>
