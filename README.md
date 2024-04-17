# Trash Prediction Server

The Trash Prediction Server is a Python-based server that takes in an image input of trash and returns a prediction of the type of trash using machine learning models.

## Introduction

This project is inspired by [this Kaggle notebook](https://www.kaggle.com/code/bhavyagiri/fastai-to-fastapi-railway/notebook), which demonstrates the integration of FastAI with FastAPI for deploying machine learning models as RESTful APIs.

## Features

- Allows users to upload images of trash via a RESTful API endpoint.
- Utilizes a trained machine learning model to predict the type of trash in the image.
- Supports multiple types of trash classification.

## Dependencies

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework for building APIs with Python.
- [FastAI](https://www.fast.ai/) - Deep learning library that simplifies training and deploying neural networks.
