[p1_q1.py] Modified version of starter.py to run on CUDA
[p1_q2.py] Modified version of starter.py to plot loss and accuracy plot
[p1_q3_inference_time.py] Added timing code to check total time for inference.
[p1_q3_training_time.py] Added timing code to check total time for training

[p2_q1.py] Modified version of simpleFC.py to plot loss and accuracy plot
[p2_q2.py] Code to draw loss plot and compare results among different probabilities for dropout by changing variable "dropout_prob" at line 46.
[p2_q3_training_time_dropout.py] Code to fill in Table2, Dropout X by adding timing code at p2_q2.py with dropout_prob=0.5
[p2_q3_traning_time_dropout_norm.py] Code to fill in Table2, Dropout X+norm by adding normalization to p2_q3_training_time_dropout.py

[p3_q1.py] Normalization added simpleCNN.py with torchsummary.
[p3_q2.py] Plotting added simpleCNN.py to test different channel size, epoch, and learning rate.
[p3_q2_dropout02.py] Enhanced version of simpleCNN by adding dropout layer between second maxpool and linear layer.
[p3_q2_L2_regularization.py] Enhanced version of simpleCNN by adding L2 normalization.
