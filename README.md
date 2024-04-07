# Assignment_2
This repository belongs to Assignment_2
 **PART - A** 
 # Libraries
I have imported all the libraries.
I have integrated wandb with projectname = 'DL_assignment_2'
# CNN class
-I have uesd the CNN inbuilt class provided by the torch library
-Initial shapes of the images are (256,256,3) 
 - The function has 5 convultion layers, 5 max pooling layers ( stride =2 ) , 5 batch normalization 
  layers, 1 dense layer and 1 output layer
- Here I am calculating the width and hight of the feature map after each convolution and maxpooling
- I have also applied drop out for overcoming the overfitting.
- **Forward function**, This function uses the activation function, choice to whether apply batch normalization and dropout and performs  feedforward operation . It return the model.
- **GPU**, I have integrated the GPU u*sing torch.device function.
- **Data Loading**, loaded the data from nature_12K.zip file. Used data transformers to reshape the data and augument the data. Depending on the function call to **datFun()** I return the tain data loader and validation data loader by either augumenting or not. I have split the entire data into 80% train data and 20% validation data.
- **training**
   - I have used the **train_fun()** which takes neurons (number of neurons in dense layer),numFilters ( number of filters ,sizeFilter (kernel size) ,activFun( activation function  ) ,optimizerName (opitimizers ) ,batchSize ( batch size) ,dropOut ( drop out) ,num_epochs ( epochs) ,learning_rate ( learning rate)  ,batchNorm ( batch normalization ) ,aug ( data augumnetation ),org (filters organization ) as the parameters.
   - Depending on the optimizer it trains the model.
   - I have used cross entropy loss values to calculate the train loss and validation loss.
   - After each batch size I update the paramets using optimizer.step()
   -  **train_fun()** calls the **check_accuracy()** function to calculate the validation accuracies and train accuracies along with cross entropy loss.
   -  


