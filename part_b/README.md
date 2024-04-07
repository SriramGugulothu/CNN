# Assignment_2
This repository belongs to Assignment_2
 **PART - B** 
 # Libraries
-I have imported all the libraries.
-I have integrated wandb with projectname = 'DL_assignment_2_B'
# wandb integration #
- I have integrated wandb to visualize the results
# Data processing #
- I have used the tronsforms class to resize and augument the data. **dataFun()** function return the appropiate data loader either augmented or normal data. Here I have split the data into 80% train and 20% validation data.
  **RESNET50(NUM_OF_CLASSES)**
  - NUM_OF_CLASSES = 10 (for nature_12k data)
  -  Changes the outpu layer neurons from 1000 to 10
  -  This function returns the model by freezing all but not last layer
  
  **RESNET50_1(k,NUM_OF_CLASSES)**
- Changes the output layer neurons from 1000 to 10
- This function returns the model by freezing first k layers only.

  **RESNET50_2(neurons_dense,NUM_OF_CLASSES)**
  - Changes the output layer neurons from 1000 to 10
  - This function returns the model by freezing all but not last layer after adding dense layer before the output layer.
    **GPU integration**
    - I have integrated the GPU using torch.device() fuction.
      **train_fun**
      - This 

