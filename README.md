# Deep learning project
### Jonatan von Martens

## Setting up

* Install requirements
    * It's important that torch and torchvision packages are up to date; I had many problems with torch 1.0.1 for instance.
* Run *run.py* to download and format data.

## Notebooks

* Notebook *lenet.ipynb* runs LeNet with downscaled 32x32 greyscale images. 

* Notebook *run_models.ipynb* runs **pretrained** AlexNet, ResNet and SqueezeNet with the tiny-imagenet images beging upscaled to 224x244.

* Notebook *run_models2.ipynb* runs **non-pretrained** AlexNet, ResNet and SqueezeNet with non-upsacled images.