# README
project-01-easy-team created by GitHub Classroom  

Train.py is used to train the model from training data.  

Test.py is used to predict labels from trained model.  

"net_params_layer1.pkl" is the parameters in layer1 from model which has been traind.  
"net_params_layer2.pkl" is the parameters in layer2 from model which has been traind.  
"net_params_linear1.pkl" is the parameters in linear1 from model which has been traind.  
"net_params_linear2.pkl" is the parameters in linear2 from model which has been traind.

To test the model with new testing data. ("Easy test") 

    replace the file name in 18th line  

    replace the file name in 20th line  
                                         
For example:  
  
    replace 'train_data = np.load('train_data.pkl')' with 'train_data = np.load('Your testing data')'  

    
    replace 'labels = np.load('finalLabelsTrain.npy')' with 'labels = np.load('Labels file')'  
             
If you want to do "Hard test", please firstly change the name of your testing file and labels in 18th and 19th line.  

Then please do as the comment said
