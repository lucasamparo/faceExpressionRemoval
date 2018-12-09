# Facial Expression Removal from 3D Images for recognition purposes
We present an encoder-decoder neural network to remove deformations caused by expressions from 3D face images. It receives a 3D face with or without expressions as input and outputs its neutral form. Our objective is not to obtain the most realistic results but to enhance the accuracy of 3D face recognition systems.

## Implementation Details

### Normalization
![Normalization](images/projection.png?raw=True "Normalization")

### Network Model
![Network Pipeline](images/pipeline.png?raw=True "Network Pipeline")
![Network Model](images/neural_net_model.png?raw=True "Network Model")

## Installation

### Requeriments
* Python 3+
* C++ 11+
* PointCloud Library (PCL)
* Tensorflow for GPU (and its dependencies)
* OpenCV
* Scipy, Scikit-learn
* RAM enough to hold all your train images

## Step-by-Step
1. Clone the repository
2. Verify the dependencies installation with install.sh
3. Execute a specific code (normalization, inference, etc...)

## Usage

### Normalize samples

### Process images

### Train with your own images

## Contributing
Pull requests are welcome. Issues request too.
Feel free to fork this project and modify whatever you need.
Please, give the credits.

Cite this work as

## License

[GNU GLPv3.0](https://choosealicense.com/licenses/gpl-3.0/)

