# blood-cell-detection

This repo has the goal to study and try some possible solutions for the blood cell detection problem. It will detect and classify the White Blood Cell (WBC), Red Blood Cell (RBC) and Platelets. The dataset used can be found in this [GitHub Repo](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset) .


## Repo structure

- **data** (after notebook run)
    - where the dataset will be saved
- **examples**
    - image & best model
- **notebooks** 
    - the study notebooks
- **yolo_data** (after notebook run)
    - reports & information related to yolo training and prediction
- other configs and scripts


## Steps to replicate

The used environment was developed with conda installed in Ubuntu in WSL2 in Windows 11.

- clone this repository
- (only on first run) conda create -n ENV_NAME python=3.11 
- conda activate ENV_NAME
- (only on first run) pip install -r requirements.txt
- open the notebook and run all commands (my findings and comments related to the development are in the notebook file)
> OBS: you will need to change the project_path variable in the first code cell of the notebook (use the absolute path)
- after that you can use the *predict.py* to see the best model predictions of a image


## Results and examples

The best model achieved can be found in *examples/best.pt* and here are some images with their classifications:

![image examples](examples/imgs_labels.png)

Some more training and evaluation information can be found in the yolo_data after the notebook is run.

## Single image prediction

It is possible to detect and predict the classes and positions of a image. To do that use:

```bash
python predict.py IMG_PATH
```
    

By default that will use the model on example, however that can be changed for others models if intended.

## comments

This was my first study into object detection, so that is one of the reasons that it is not as deep. However I'll keep trying best techniques and improve bit by bit. For suggestions and other comments, please contact me.







