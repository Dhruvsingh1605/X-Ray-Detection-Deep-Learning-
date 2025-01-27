# Pneumonia Detection from X-Ray Images

This project uses a Convolutional Neural Network (CNN) based on the **VGG19** architecture to classify X-ray images as either `Normal` or `Pneumonia`. The model was trained on a dataset of chest X-rays and deployed using **Streamlit** for an interactive user interface.

---

## Features

- **Deep Learning Model**: Utilizes a pre-trained VGG19 model fine-tuned for pneumonia detection.
- **Streamlit Interface**: Easy-to-use web interface for uploading and analyzing X-ray images.
- **Real-Time Predictions**: Upload an image, and the app will classify it as `Normal` or `Pneumonia` with a probability score.

---

## Project Structure

```plaintext
├── app.py                 # Main Streamlit app script
├── vgg_unfrozen.h5        # Pre-trained model weights
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies


## Clone the repository

git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection

## Create an Environment

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

## RUn the Streamlit App
streamlit run app.py

## Dependencies
The following Python libraries are required:

TensorFlow
Keras
Streamlit
NumPy
OpenCV
Pillow
h5py

git 