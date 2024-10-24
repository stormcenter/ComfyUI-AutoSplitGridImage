# ComfyUI-AutoSplitGridImage

ComfyUI-AutoSplitGridImage is a custom node for ComfyUI that provides flexible image splitting functionality. It allows users to choose between edge detection and uniform division for both row and column splits, offering a customizable approach to grid-based image segmentation.

![example_workflow](./2024-10-23_10-59-03.png)

![example_png](./ComfyUI_temp_zbacb_00001_.png)

## Features

- Customizable splitting methods for both rows and columns
- Choice between edge detection and uniform division for each axis
- Adjustable number of rows and columns
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
4. Choose the splitting method for rows and columns (uniform or edge detection).
5. The node will output two images:
   - A preview image showing the grid lines
   - A tensor containing all the split image cells

## Parameters

- `image`: Input image to be split
- `rows`: Number of rows to split the image into (default: 2, range: 1-10)
- `cols`: Number of columns to split the image into (default: 3, range: 1-10)
- `row_split_method`: Method to split rows ("uniform" or "edge_detection")
- `col_split_method`: Method to split columns ("uniform" or "edge_detection")

## How It Works

- Uniform Splitting: Divides the image into equal parts along the specified axis.
- Edge Detection Splitting: Uses OpenCV's Canny edge detection to find natural splitting points in the image.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project is designed to work with [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- Edge detection is implemented using OpenCV.
