# iNaturalist 2019 pipeline
This pipeline will generate and publish MobilenetV2 model ready to predict trained on iNaturalist 2019 Dataset (Plants only).

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
A file called `inat-2019.tar.gz` will be created. This is the pipeline file you should upload into Kubeflow UI.
After the upload, the pipeline is ready to go!