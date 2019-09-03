# CIFAR-10 pipeline
This pipeline will generate and publish a CNN model ready to predict trained on CIFAR-10 Dataset.

## Data handling
A notebook called `data-preparation.ipynb` should be played to get initial tfrecords from the entire dataset.
This is mandatory before running any pipeline version.

## Interactive Jupyter notebook
A notebook called `interactive.ipynb` with the entire pipeline and data handling visualizations is included.
It can be served in a notebook instance by Kubeflow mounting the `tfx` PersistentVolume.

## Build and launch
To build the pipeline is:
```
python pipeline.py
```
A file called `cifar-10.tar.gz` will be created. This is the pipeline file you should upload into Kubeflow UI.
After the upload, the pipeline is ready to go!