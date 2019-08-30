# TFX Kubeflow pipelines
Kubeflow pipelines built on top of Tensorflow TFX library

## General info
This repository contains machine learning pipelines based on Tensorflow TFX library.
Every pipeline is designed to be published on a Kubernetes/Kubeflow cluster *on premise*.

### Tutorials
Each folder contains needed code and data for the Kubeflow Pipeline, plus a README that includes:

* pipeline general information
* specific Kubernetes resources setup instructions
* specific data handling about pipeline on premise
* build and launch procedure

Further pipelines are welcome via pull request.

## Pipelines:
* **[iris](iris)** - Complete pipeline for a DNN model on IRIS dataset.

#### Useful links
* [Kubeflow](https://www.kubeflow.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Tensorflow TFX](https://www.tensorflow.org/tfx)
