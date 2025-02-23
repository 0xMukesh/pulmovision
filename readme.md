# bio-med

a biomedical image classifier built on top of resnet and uses [medMNIST](https://medmnist.com/) as the dataset

## usage

1. install the dependencies using the following command

   ```
   pip install -r requirements.txt
   ```

2. train the model by running `python main.py`

3. start streamlit server by running `streamlit run streamlit.py`

## f1 scores

### vanilla CNN

```
micro-f1: 0.11  macro-f1: 0.087

per-class f1 scores:

Atelectasis: 0.215
Cardiomegaly: 0.071
Effusion: 0.085
Infiltration: 0.034
Mass: 0.096
Nodule: 0.112
Pneumonia: 0.021
Pneumothorax: 0.093
Consolidation: 0.082
Edema: 0.046
Emphysema: 0.044
Fibrosis: 0.032
Pleural: 0.000
Hernia: 0.01
```

### resnet

(have implemented the logic but couldn't get the time to complete the training)
