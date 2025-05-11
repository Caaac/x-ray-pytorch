# Dataset info
## Analysis of X-ray images based on pictures with disease prediction

### [Info about dataset's classes](https://huggingface.co/datasets/Sohaibsoussi/NIH-Chest-X-ray-dataset-small)

```py
class_label:
  '0': No Finding
  '1': Atelectasis
  '2': Cardiomegaly
  '3': Effusion
  '4': Infiltration
  '5': Mass
  '6': Nodule
  '7': Pneumonia
  '8': Pneumothorax
  '9': Consolidation
  '10': Edema
  '11': Emphysema
  '12': Fibrosis
  '13': Pleural_Thickening
  '14': Hernia
```



# Result of training

## Densenet121

### №1


#### Train data

* <b>Loss:</b> CrossEntropyLoss
* <b>Optimizer:</b>
  - <b>Type:</b> Adam
  - <b>Learning rate:</b> 1e-3
  - <b>L2:</b> 1e-3
* <b>Transform:</b>
  1. ToTensor()
  1. ToDtype(dtype=torch.float32, scale=True)
* <b>CNN:</b> Densenet121 (requires_grad = False)
* <b>FCL:</b> 
  ```py
  nn.Sequential(
      nn.Linear(1024, 2048),
      nn.ReLU(inplace=True),
      nn.Dropout(.25),
      nn.Linear(2048, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(.25),
      nn.Linear(512, 15)
  )
  ```

#### Accurace and Loss by epochs

![alt text](./assets/readme/image-1.png)

#### ROC Curves

![alt text](./assets/readme/image.png)


### №2


#### Train data

* <b>Loss:</b> CrossEntropyLoss
* <b>Optimizer:</b>
  - <b>Type:</b> Adam
  - <b>Learning rate:</b> 1e-3
  - <b>L2:</b> 1e-3
* <b>Transform:</b>
  1. ToTensor()
  1. ToDtype(dtype=torch.float32, scale=True)
* <b>CNN:</b> Densenet121 (requires_grad = True)
* <b>FCL:</b> 
  ```py
  nn.Sequential(
      nn.Linear(1024, 2048),
      nn.ReLU(inplace=True),
      nn.Dropout(.25),
      nn.Linear(2048, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(.25),
      nn.Linear(512, 15)
  )
  ```

#### ROC Curves

![alt text](./assets/readme/2-image.png)

### №3 (Один Dropout)


#### Train data

* <b>Loss:</b> CrossEntropyLoss
* <b>Optimizer:</b>
  - <b>Type:</b> Adam
  - <b>Learning rate:</b> 1e-3
  - <b>L2:</b> 1e-3
* <b>Transform:</b>
  1. ToTensor()
  1. ToDtype(dtype=torch.float32, scale=True)
* <b>CNN:</b> Densenet121
* <b>FCL:</b> 
  ```py
  nn.Sequential(
      nn.Linear(1024, 2048),
      nn.ReLU(inplace=True),
      nn.Dropout(.25),
      nn.Linear(2048, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 15)
  )
  ```

Test Average Precision: 0.0197
Test Average Recall: 0.0667
Test Average F1-score: 0.0304

#### ROC Curves

![alt text](./assets/readme/3-1-image.png)


### №4 (Without Dropout)


#### Train data

* <b>Loss:</b> CrossEntropyLoss
* <b>Optimizer:</b>
  - <b>Type:</b> Adam
  - <b>Learning rate:</b> 1e-3
  - <b>L2:</b> 1e-3
* <b>Transform:</b>
  1. ToTensor()
  1. ToDtype(dtype=torch.float32, scale=True)
* <b>CNN:</b> Densenet121
* <b>FCL:</b> 
  ```py
  nn.Sequential(
      nn.Linear(1024, 2048),
      nn.ReLU(inplace=True),
      nn.Linear(2048, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 15)
  )
  ```

#### Macro scores

* Test Average Precision: 0.0197
* Test Average Recall: 0.0667
* Test Average F1-score: 0.0304

#### ROC Curves

![alt text](./assets/readme/4-1-image.png)



## Densenet121 Multi-Class

### №1


#### Train data

* <b>Loss:</b> BCEWithLogitsLoss
* <b>Optimizer:</b>
  - <b>Type:</b> Adam
  - <b>Learning rate:</b> 1e-4
  - <b>L2:</b> 1e-4
* <b>Transform:</b>
  1. ToTensor()
  1. ToDtype(dtype=torch.float32, scale=True)
* <b>CNN:</b> Densenet121 (requires_grad = False)
* <b>FCL:</b> 
  ```py
  nn.Sequential(
      nn.Linear(1024, 15),
  )
  ```

#### Macro scores

* Test Average Precision: 0.0282
* Test Average Recall: 0.0667
* Test Average F1-score: 0.0396
<!-- 
<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th>Number Class</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>0.42</td><td>1.0</td><td>0.59</td><td>541</td></tr>
    <tr><td>1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>160</td></tr>
    <tr><td>2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30</td></tr>
    <tr><td>3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>133</td></tr>
    <tr><td>4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>190</td></tr>
    <tr><td>5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>37</td></tr>
    <tr><td>6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>31</td></tr>
    <tr><td>7</td><td>0.0</td><td>0.0</td><td>0.0</td><td>4</td></tr>
    <tr><td>8</td><td>0.0</td><td>0.0</td><td>0.0</td><td>79</td></tr>
    <tr><td>9</td><td>0.0</td><td>0.0</td><td>0.0</td><td>27</td></tr>
    <tr><td>10</td><td>0.0</td><td>0.0</td><td>0.0</td><td>4</td></tr>
    <tr><td>11</td><td>0.0</td><td>0.0</td><td>0.0</td><td>15</td></tr>
    <tr><td>12</td><td>0.0</td><td>0.0</td><td>0.0</td><td>15</td></tr>
    <tr><td>13</td><td>0.0</td><td>0.0</td><td>0.0</td><td>12</td></tr>
    <tr><td>14</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1</td></tr>
  </tbody>
</table> -->

#### Accurace and Loss by epochs

![alt text](./assets/readme/d121-ml-1-1-image.png)

#### ROC Curves

![alt text](./assets/readme/d121-ml-1-2-image.png)

