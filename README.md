# CNN Building Blocks: Convolution, Activation, and Pooling

A hands-on Jupyter notebook that shows, step by step, how a convolutional neural network (CNN) processes an image. It uses TensorFlow to apply the core building blocks of a CNN layer to a single grayscale image and visualizes the result at each stage.

## What it covers

| Section | What it demonstrates |
|---|---|
| Load and prepare the image | Decode a JPEG as grayscale, resize to 300x300, convert to `float32`, and add a batch dimension to get the shape `(1, 300, 300, 1)` |
| Convolution | Apply a 3x3 edge-detection kernel with `tf.nn.conv2d` and view the output with the magma colormap |
| ReLU and max pooling | Apply `tf.nn.relu`, then 2x2 max pooling, and compare the three stages side by side |
| Kernel experiments | Compare Sobel X, Sobel Y, box blur, and sharpen kernels using a reusable `apply_kernel(k)` function |
| Padding, stride, and pooling shapes | Predict output shapes from the SAME and VALID formulas, then verify them in code |

## The pipeline

```
Image  ->  Convolution  ->  ReLU  ->  Max pooling
(1,300,300,1)  (1,300,300,1)  (1,300,300,1)  (1,150,150,1)
```

- **Convolution** slides a small kernel over the image to detect features such as edges.
- **ReLU** sets negative values to zero.
- **Max pooling** keeps the strongest value in each 2x2 window, reducing the number of values by a factor of 4.



| Kernel | Matrix | Effect |
|---|---|---|
| Edge detection | `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` | Highlights where brightness changes; weights sum to 0 |
| Sobel X | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | Responds to vertical edges |
| Sobel Y | `[[-1,-2,-1],[0,0,0],[1,2,1]]` | Responds to horizontal edges |
| Box blur | `[[1,1,1],[1,1,1],[1,1,1]] / 9` | Smooths by averaging neighbors |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | Increases local contrast |



For input size `n`, kernel or window size `k`, and stride `s`:

- **SAME** padding: `ceil(n / s)`
- **VALID** padding: `floor((n - k) / s) + 1`

Example for a 300x300 input with a 3x3 kernel:

| Operation | Output shape |
|---|---|
| Convolution, stride 1, SAME | `(1, 300, 300, 1)` |
| Convolution, stride 1, VALID | `(1, 298, 298, 1)` |
| Convolution, stride 2, SAME | `(1, 150, 150, 1)` |
| Max pool 2x2, stride 2, SAME | `(1, 150, 150, 1)` |
| Max pool 4x4, stride 4, SAME | `(1, 75, 75, 1)` |
| Convolution VALID, then max pool 2x2, stride 2 | `(1, 149, 149, 1)` |

## Requirements

- Python 3.11 (a TensorFlow-supported version)
- TensorFlow
- Matplotlib
- Jupyter (or VS Code with the Python and Jupyter extensions)

## Setup

1. Clone or download this repository.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install tensorflow matplotlib jupyter
   ```

## Usage

1. Open the notebook in VS Code or Jupyter.
2. Select the `.venv` Python interpreter as the kernel.
3. Run the cells from top to bottom (**Run All**). Later cells reuse the image tensor from the first section, so the order matters.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `NotFoundError` or `FileNotFoundError` when reading the image | The image is not in the working folder, or the name differs | Check with `import os; print(os.listdir())` and fix the path or name |
| Error from `decode_jpeg` | The file is not a real JPEG (for example, a renamed PNG) | Use a true JPEG, or switch to `tf.io.decode_image` |
| `NameError` | Cells were run out of order | Restart the kernel and run all cells from the top |
| Image appears in purple and yellow instead of gray | Missing `cmap="gray"` | Add `cmap="gray"` to `plt.imshow` |

## Key takeaways

- A kernel's weights decide what the layer detects.
- Kernels whose weights sum to 0 give about 0 on flat regions, so only edges appear. Kernels that sum to 1 preserve overall brightness.
- SAME padding with stride 1 keeps the image size; VALID shrinks it but avoids artificial zero borders.
- Pooling reduces computation and tolerates small shifts, at the cost of fine detail.
- In a real CNN, kernel values are learned during training rather than chosen by hand.
