# Notes for first running  this project

## Installtion (using conda)

- create env by using ".yml" file
- if pip install 'clip' and 'kaolin' failed
- install git first and check the right version for kaolin
```
conda install git
```
```
# Replace TORCH_VERSION and CUDA_VERSION with your torch / cuda versions
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VERSION}_cu{CUDA_VERSION}.html
```
- using the [link](https://pytorch.org/get-started/previous-versions/) to check the right version
- using the command in demo