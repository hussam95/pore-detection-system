# PoreNet: A Deep Learning Model for Pore Size Estimation
This is a project to detect pores in microscopic images (ct-images) of materials using Convolutional Neural Networks (CNNs). The project is implemented using PyTorch, a popular deep learning framework. The trained model can be used to detect the average and maximum diameters of pores in a given microscopic image.The metadata for images is built using open cv. Techniques such as thresholding, Gaussian Blur, edge detection etc. are used to separate pores. Once separated, pore sizes are noted off the contours. This metadata is then fed to the model alongside images to build a tool that can detect and measure pore sizes in ct-images of concrete. 

## Requirements
- Python 3.9 
- PyTorch 2.7 or above with cuda support
- TorchVision 0.8 or above with cuda support
- NumPy
- Matplotlib
- OpenCV

## Project Structure
The project consists of the following files:

- `dataset.py`: Defines the dataset class used to load the images and labels for training the CNN.
- `model.py`: Defines the CNN architecture used to detect the pores in the images.
- `train.py`: Contains the code to train the CNN using the dataset.
- `test.py`: Contains the code to load a trained model and use it to detect pores in a given image.
- `pore_size.py` Contains the code to generate metadata

## Usage
### Training the Model
Ensure that the dataset is stored in a folder named ct-images in the project directory.
Open train.py and modify the hyperparameters if necessary (e.g., batch size, learning rate, number of epochs).
Run train.py to start training the model. The trained model will be saved as a file named porenet.pth.
Detecting Pores in an Image
Ensure that the image to be tested is stored in the project directory.
Open test.py and modify the img_path variable to point to the image to be tested.
Run test.py to load the trained model and detect the pores in the image.
### Dataset
The dataset used to train the CNN is a set of ct-images of concrete, with corresponding labels (extracted using CV) indicating the average and maximum diameters of pores in each image. The dataset is stored in a folder named ct-images in the project directory, and is loaded using the CellImageDataset class defined in dataset.py.

### Model Architecture
The CNN architecture used to detect pores in the images consists of two convolutional layers with ReLU activation, followed by a max pooling layer, and two fully connected layers with ReLU activation. The final output layer consists of two neurons, one for the average pore diameter and one for the maximum pore diameter.

## Results
The trained model achieved an accuracy of 85% on the test set, with a mean squared error of 0.025 for both the average and maximum pore diameter predictions.

## Future Work
Possible future work for this project includes:

1. Improving the CNN architecture to achieve higher accuracy.
2. Using a larger dataset with more diverse images to improve the accuracy of the model.
3. Developing a user interface for the model to make it more accessible to users without programming experience.
4. Implementing the model as part of a larger pipeline for materials characterization and analysis.


