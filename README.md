# kuwuhara

An implementation of [Anisotropic Kuwahara Filtering](https://www.umsl.edu/~kangh/Papers/kang-tpcg2010.pdfhttps://www.umsl.edu/~kangh/Papers/kang-tpcg2010.pdf) with CUDA.

## Usage

In order to use this project, you will require [CUDA](https://developer.nvidia.com/cuda-toolkit) installed on your computer.
```sh
mkdir out && cd out
cmake ..
# If on Linux
make
# Or on Windows
ninja
```

The program expects a input file `input.png` in the same directory as the executable. After running, it will produce a `output.png`, with the filter applied.
