# Convolution Neural Network part -1 
This repository belongs to Deep Learning course project.
 **PART - A** 
 # Libraries
I have imported all the libraries.
# wandb
I have integrated wandb with projectname = 'DL_Project_1'
# CNN class
-I have uesd the CNN inbuilt class provided by the torch library
-Initial shapes of the images are (256,256,3) 
 - The function has 5 convulation layers, 5 max pooling layers ( stride =2 ) , 5 batch normalization 
  layers, 1 dense layer and 1 output layer
- Here I am calculating the width and hight of the feature map after each convolution and maxpooling
- I have also applied drop out for overcoming the overfitting.
- **forward(x) function**, This function uses the activation function, choice to whether apply batch normalization, dropout and performs feedforward operation . It return the model.
- **GPU**, I have integrated the GPU using torch.device function.
- **Data Loading**, loaded the data from nature_12K.zip file. Used data transformers to reshape the data and augument the data. Depending on the function call to **datFun()** I return the tain data loader and validation data loader by either augumenting or not. I have split the entire data into 80% train data and 20% validation data.
- **training**
   - I have used the **train_fun()** which takes neurons (number of neurons in dense layer),numFilters ( number of filters ,sizeFilter (kernel size) ,activFun( activation function  ) ,optimizerName (opitimizers ) ,batchSize ( batch size) ,dropOut ( drop out) ,num_epochs ( epochs) ,learning_rate ( learning rate)  ,batchNorm ( batch normalization ) ,aug ( data augumnetation ),org (filters organization ) as the parameters.
   - Depending on the optimizer it trains the model.
   - I have used cross entropy loss values to calculate the train loss and validation loss.
   - After each batch size I update the paramets using optimizer.step()
   -  **train_fun()** calls the **check_accuracy()** function to calculate the validation accuracies and train accuracies along with cross entropy loss at each epoch. They are posted into wandb.
   -  Here sometimes I didn't include the parameters inside the function if they are more in number. Mostly explained their functionalities 
**chek_accuracy(loader,model,criterion,batchSize)**
- This function calculates the accuracies and losses of the train and validation data depending on the train loader or validation loader passed to it
  
   **trainPartA.py**

  -- I have used the parse_arguments from parser library to execute thie trainPartA.py file.
  -- It can be executed by appling !python trainPartA.py --(parameters that are supported as choices in my trainPartA.py file)
  -- (**-- parameterName**) command to test with other values than default values can be used.
  
  **dl_project_1.ipynb.**
  -- sorry for the mistake I did not mention the part name in the file name.
  -- In place of parsers I have integrated with wandb parameters and ran the sweeps in ipynb file.
  --  You can run by integratin your wandb account by activation key to visualize the accuarcies.


