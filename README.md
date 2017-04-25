# selfDrivingCar
supervised Self Driving Car using Udacity simulator

Link to Udacity Simulator setup : https://github.com/udacity/self-driving-car-sim
Follow the steps in the github repository of udacity simulator to setup the environment.

Steps to run the simulator from dirve.py:

Open a terminal, then execute the following command to clone the project to your computer.
  > git clone https://github.com/gvp-raviteja/selfDrivingCar

Create a python environment used by the model using conda:
  > cd car-behavioral-cloning
  > conda env create --name

Activate the python environment used by the model
  > source activate car-behavioral-cloning

Start the simulator by running the executable and start the game in Autonomous mode

Go back to the terminal and run the pre-trained model by the following command:

  > python drive.py model.h5
  
  
Drive.py acts as an interface between the simulator and CNN supervised model.

Model.py is used to train the CNN model on the given dataset.
