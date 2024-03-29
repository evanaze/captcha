![Sample Captcha](data/raw/1_f676409e63c94448a5903a80e9da114f.png)

# Captcha Solver and Object Counting
This is a Python package for solving captchas that look like the one above.

## Usage
Pull the repo with the usual command:  

```git clone https://github.com/evanaze/captcha.git && cd captcha```  

You can download the data for this project from S3 with:  

```make data```

And train the model with 

```make train```

To use the to predict on a new image, run with the `-i` flag:  

```python -m src.run -i new_image.png```  

Alternatively, to run by retraining on new data with a learning rate of `0.4`:  

```python -m src.run --retrain --save-model --lr 0.4```  
or 
```make train``` to use the default settings.

## Requirements
You just need [Docker](https://www.docker.com) to run this package. If you would like to run this package inside of a local environment, you need Python3 and Pytorch installed. 