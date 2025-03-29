# GoClassify

I have utilized a **PyTorch-trained image classification model** and the **MNIST dataset** in this project to classify handwritten digits. The **PyTorch-trained** model is hosted via a **REST API** over the **Gin web framework** in **Go**.

## Project Overview

This is a sample of **machine learning** with **Go**. I employed a basic **fully connected neural network (FC-512-256-10)** and trained the model to recognize handwritten digits (0-9). The trained model is executed as a REST API where clients can provide images for recognition.

- **Dataset**: The MNIST database of 28x28 grayscale handwritten digits (0-9).
- **Model Architecture**: A dense neural network with an FC-512-256-10 architecture, where all hidden layers use the ReLU activation function, and the output layer uses the softmax activation function.
- **Frameworks**:
  - **PyTorch**: Used for training and validating the model.
  - **Go (Gin)**: Used to implement and serve the REST API to interact with the trained model.

## Features

- **Handwritten Digit Prediction**: The model can predict a digit (0-9) from an image of a handwritten digit.
- **API**: I created an API using the Gin framework, allowing users to upload an image file and receive the predicted digit along with class probabilities.
- **Model Serving**: The trained model is saved and served as a REST API for easy and convenient access.

## API Endpoint

- **POST /classify**
  - **Description**: This API endpoint classifies a handwritten digit image and returns the predicted digit and probabilities for each digit.
  - **Request**: A POST request with an image file, with the field name `image`.
  - **Response**:
    ```json
    {
      "predicted_digit": 5,
      "probabilities": {
        "0": 0.02,
        "1": 0.03,
        "2": 0.04,
        "3": 0.05,
        "4": 0.06,
        "5": 0.65,
        "6": 0.01,
        "7": 0.02,
        "8": 0.03,
        "9": 0.05
      }
    }
    ```

## Model Structure

- **Input**: The model was trained on 28x28 grayscale images of handwritten digits.
- **Network**: The network consists of the following layers:
  - **FC-512**: The first fully connected layer with 512 neurons.
  - **FC-256**: The second fully connected layer with 256 neurons.
  - **FC-10**: The output layer with 10 neurons, each corresponding to a digit from 0 to 9.
  - **Activation**: ReLU activation for all hidden layers.
  - **Output**: Softmax activation on the final layer to produce the final probability distribution over the digits.

## How I Built the Project

1. **Data Preprocessing**: Loaded the MNIST dataset from PyTorch's pre-defined data loaders and applied normalization and resizing to ensure the images were 28x28 pixels.
2. **Model Training**: Trained the model using a fully connected neural network. Cross-entropy loss was used as the loss function, and the Adam optimizer was applied. The model was trained for 10 epochs.
3. **Model Evaluation**: I tested the model's performance using a test set to evaluate accuracy and loss. I also generated prediction plots to assess whether the model was performing well.
4. **Serving the Model**: After training, I saved the model's weights and served the trained model through an API based on the Gin framework. The API accepts image uploads and returns the predicted digit and class probabilities as a JSON response.

## Dependencies

### Python
- `torch` (PyTorch)
- `torchvision` (For the MNIST dataset)
- `PIL` (For image processing)

### Go
- `github.com/gin-gonic/gin` (For serving the API)

## License

MIT License.
