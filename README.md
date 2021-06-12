This repo could be installed and run 
$python3 ./examples/example1.py
Drawing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90/90 [00:01<00:00, 64.04it/s]


## My nvcc version(in linux)
$ nvcc -V
nvidia-smi :nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Apr_24_19:10:27_PDT_2019
Cuda compilation tools, release 10.1, V10.1.168

## Other lib version in requirements.txt
## Nvidia Driver version (in linux)
$ nvidia-smi
Sat Jun 12 17:19:41 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:07.0 Off |                    0 |
| N/A   31C    P0    39W / 300W |      0MiB / 16160MiB |      0%      Default |
..........
..........
..........





# Neural 3D Mesh Renderer (CVPR 2018)

This repo contains a PyTorch implementation of the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.
It is a port of the [original Chainer implementation](https://github.com/hiroharu-kato/neural_renderer) released by the authors.
Currently the API is the same as in the original implementation with some smalls additions (e.g. render using a general 3x4 camera matrix, lens distortion coefficients etc.). However it is possible that it will change in the future.

The library is fully functional and it passes all the test cases supplied by the authors of the original library.
Detailed documentation will be added in the near future.
## Requirements
Python 2.7+ and PyTorch 0.4.0.

The code has been tested only with PyTorch 0.4.0, there are no guarantees that it is compatible with older versions.
Currently the library has both Python 3 and Python 2 support.

**Note**: In some newer PyTorch versions you might see some compilation errors involving AT_ASSERT. In these cases you can use the version of the code that is in the branch *at_assert_fix*. These changes will be merged into master in the near future.
## Installation
You can install the package by running
```
pip install neural_renderer_pytorch
```
Since running install.py requires PyTorch, make sure to install PyTorch before running the above command.
## Running examples
```
python ./examples/example1.py
python ./examples/example2.py
python ./examples/example3.py
python ./examples/example4.py
```


## Example 1: Drawing an object from multiple viewpoints

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example1.gif)

## Example 2: Optimizing vertices

Transforming the silhouette of a teapot into a rectangle. The loss function is the difference between the rendered image and the reference image.

Reference image, optimization, and the result.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_optimization.gif) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_result.gif)

## Example 3: Optimizing textures

Matching the color of a teapot with a reference image.

Reference image, result.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example3_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example3_result.gif)

## Example 4: Finding camera parameters

The derivative of images with respect to camera pose can be computed through this renderer. In this example the position of the camera is optimized by gradient descent.

From left to right: reference image, initial state, and optimization process.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example4_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example4_init.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example4_result.gif)


## Citation

```
@InProceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
