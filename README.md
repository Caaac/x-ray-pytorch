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