# ComfyUI-AutoSplitGridImage

ComfyUI-AutoSplitGridImage is a custom node for ComfyUI that provides intelligent image splitting functionality. It combines edge detection for column splits and uniform division for row splits, offering a balanced approach to grid-based image segmentation.

![example_workflow](./2024-10-23_10-59-03.png)

![example_png](./ComfyUI_temp_zbacb_00001_.png)

## Features

- Intelligent column splitting using edge detection
- Uniform row splitting for consistent horizontal divisions
- Customizable number of rows and columns
- Preview image with grid lines
- Outputs both the preview image and individual grid cells

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-AutoSplitGridImage.git
   ```
2. Restart ComfyUI or reload custom nodes.

## Usage

1. In the ComfyUI interface, find the "GridImageSplitter" node under the "image/processing" category.
2. Connect an image output to the "image" input of the GridImageSplitter node.
3. Set the desired number of rows and columns.
4. The node will output two images:
   - A preview image showing the grid lines
   - A tensor containing all the split image cells

## Parameters

- `image`: Input image to be split
- `rows`: Number of rows to split the image into (default: 2, range: 1-10)
- `cols`: Number of columns to split the image into (default: 3, range: 1-10)

## How It Works

- Column Splitting: Uses edge detection to find natural splitting points in the image.
- Row Splitting: Applies uniform division for consistent horizontal splits.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project is designed to work with [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- Edge detection is implemented using OpenCV.
