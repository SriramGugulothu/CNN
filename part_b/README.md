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
    **Training**
      - I have used the funciton **train_fun(batchSize,num_epochs,learning_rate,aug,strategy,NUM_OF_CLASSES)** for training. batch size (batchSize),epochs (num_epochs),learning rate (learning_rate),augumentation (aug),strategy ,NUM_OF_CLASSES = 10
      - Here strategy = 0 ,model by freezes all but not last layer
      - strategy =1, model by freezes first k layers only
      - strategy = 2, model by freezing all but not last layer after adding dense layer
      This function uses the adam as optimizer and fill the training accuracy and validation accuracy along with losses in wandb. It uses the check_accuracy function to get the accracies and losses.
**chek_accuracy(loader,model,criterion,batchSize)**
- this function calculates the accuracies and losses of the train and validation data depending on the train loader or validation loader passed to it.
  
**trainPartB.py**
  -- I have used the parse_arguments from parse library to execute thie trainPartB.py file.
  -- It can be execute by appling !python trainPartB.py --(parameters that are supported as choices in my trainB.py file)
  -- (**-- parameterName**) command to test with other values than default values can be used.
 **dl_assignment_2_partb.ipynb.**
  -- In place of parsers I have integrated with wandb parameters and ran the sweeps in ipynb file.
  -- You use your wandb api key to visualize the accuracies aand losses in wandb.


