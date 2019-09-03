# TFX Kubeflow pipelines
Kubeflow pipelines built on top of Tensorflow TFX library

## General info
This repository contains machine learning pipelines based on Tensorflow TFX library.
Every pipeline is designed to be published on a Kubernetes/Kubeflow cluster *on premise*.

Each folder contains needed code and data for the Kubeflow Pipeline, plus a README that includes:

* pipeline general information
* specific data handling about pipeline on premise
* interactive notebooks instructions
* build and launch procedure

Further pipelines are welcome via pull request.

## Pipelines:
* **[iris](iris)** - Complete pipeline for a DNN model on IRIS dataset.
* **[cifar-10](cifar-10)** - Complete pipeline for a CNN model on CIFAR-10 dataset.
* **[inat-2019](inat-2019)** - Complete pipeline for a MobilenetV2 model on iNaturalist 2019 dataset.

## Prerequisites
Here some prerequisites needed to deploy this repo.

### Platform versions
* Kubeflow version >=0.6.2
* Tensorflow >=1.14.0 <2.0
* Tensorflow TFX ==0.14.0rc1

### Kubernetes cluster
A PersistentVolumeClaim called `tfx-pvc` is needed so the cluster should have one ready *before* dropping the pipelines.

Here an example of a 100Gb claim with a local-path storageClass onboard.
```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tfx-pvc
  namespace: kubeflow
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-path
  resources:
    requests:
      storage: 100Gi
```

### Utils files deployment
Cloning this repository into the root of the `tfx` PersistentVolume is needed before starting any pipeline.

## Local development and building
Some python libraries are needed. Install them with:
```
pip install -r requirements.txt
```
requirements.txt file is on root of this repo.

## Useful links
* [Kubeflow](https://www.kubeflow.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Tensorflow TFX](https://www.tensorflow.org/tfx)
