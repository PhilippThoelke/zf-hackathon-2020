# ZF Closed Loop Hackathon 2020: Team RandomBaseline
## The Problem
The task is to find a way to optimally control the current (parameter: i) applied to an active damping system in order to optimize comfort and safety.


## Basic Concept
Since a solver is slow and complicated we use to power of artificial neural networks to predict the current for each time step. However, the objective function that is given cannot be evaluated over a single time step due to its dependency on the variance. Thus, backpropagation cannot be easily applied.
Consequently, we do not directly train our networks but introduce a genetic algorithm which selects well performing networks of a population after a certain amount of time steps. These selected networks are preserved and the bad performing networks get replaced with mutations and crossovers of the good performing ones.
After training, the best performing network should be able to serve as a control system, predicting a well chosen value for the current.

## File Descriptions
### dataProcessing
This file contains the ProfileManager class which is responsible for handling the datasets.

Here the .csv files are read in and and the road profiles are interpolated to fit a constant velocity. We store each profile for different velocities in a list.
In order to improve training time we store the ProfileManager in a pickle file which can be read in, meaning we need to only once load all data in order to train for it with different setups. Thus, running the script loads all data and stores the ProfileManager in a pickle. Once this pickle exists, this sript does not need to be executed again since the geneticAlgorithm directly uses the pickle file to load the road profiles.

### environment
This script simulates our environment with which our neural network interacts.

It contains the class Simulator which handles the states.
With 'Simulator.next(i_new)' a new state is computed using 'Simulator.active_suspension()' which takes all the parameters of a current state and the prediction for the current in order to compute a new state. This new state is then scored in the owned list of states and can be used to pass on to the neural network for the next training step.
The Simulator also computes the value of the objective function 'Simulator.t_target()' and checks if the constraints given by 'Simulator.constraint_satisfied()' are met in order to compute a final score which can be used as a fitness function to select well performing neural networks. All this is done by calling 'Simulator.score()'

A states is given by the following 9-tuple: (Zb, Zb_dt, Zb_dtdt, Zt, Zt_dt, Zt_dtdt, i, Zh, Zh_dt)

* Zb: z-position body [m]
* Zt: z-position tire [m]
* Zh: road profie [m]
* Zb_dt: velocity body in z [m/s]
* Zt_dt: velocity tire in z [m/s]
* Zh_dt: velocity road profile in z [m/s]
* ...
* x_dtdt: specific acceleration in [m/s^2]
* i: current of active suspension from 0 to 2 [A]

### geneticAlgorithm
This is the heart of the whole project and has to be executed in order to train a model.

In the ANN class the neural network is defined using the framework *Pytorch*. The framework is similar to *Tensorflow 2.x* but easier to use in combination with multithreading which is needed to speed up the training process.  

The GeneticAlgorithm class creates neural networks, use them to predict the current and also handles the selection and alternations of the networks.
To train, we choose a population of networks. We then choose a random road profile and a random position on this road profile. From there we let the single networks perform a certain number of steps before a population gets evaluated and updated. Then this whole process is repeated with the updated population. During this process we sometimes store the best performing network in order to use later for evaluation.

### evaluator
This script takes a ready trained model and lets it run on a data set, storing all the parameters including the prediction of the current (parameter: i) in a csv. file. This can also be done using a constant value for i.
We are also able to read in csv including a predicted i and evaluating how well this prediction is according to our target and constraints.



## hyperparameters
This file contains all hyperparameters used.
The important ones are:
* EPOCHS: in an epoch a population is evaluated and updated
* EVALUATION_REPEATS: how many road position we look at for each epoch
* EVALUATION_STEPS:  how many steps we train from the picked road position onward
* POPULATION_SIZE: number of nets in a generation
* NUM_SURVIVORS: number of best nets which remain unchanged
* MUTATION_RATE: how strongly the weights get altered when mutating
* MUTATION_SCALE
* VEL: a list of different velocities we train on
* K: a list of different k values we train on


## Training
We trained different alternations to evaluate later:

### Default
We normally trained on all data sets except "" and used 3 different velocities namely 8, 20 and 27 m/s. The default value for k was 3.
The commonly used values for the genetic algorithm were:
- EPOCHS = 1000
- EVALUATION_REPEATS = 10
- EVALUATION_STEPS = 1000
- POPULATION_SIZE = 96
- NUM_SURVIVORS = 18
- MUTATION_RATE = 0.015
- MUTATION_SCALE = 0.3

### Dense vs. Convolution
We used different neural network architectures where we either only used dense layers combined with sigmoid activation functions or convolutional layers. In the convolutional networks we also tried passing on more than one single state to the network but a certain number of previous states.

### Single State vs. Running Average
Instead of just passing on a single state to the neural networks we tried passing on two states where one represents the running average over all previous states.

### Architectures
We mainly tried out two different architectures for the networks, one with only one hidden layer and one with two.
1. Input: 9, Hidden: 8 4 Output: 1
2. Input: 9, Hidden: 8, Output: 1

## Evaluation & Findings
- convolution deos not make a difference
- training on randomly chosen ks disrupt the training process: it seems to be a better idea to train a network for each k and than select which net is needed when performing
- on the initial data a constant i of 0 outperformed all of our trained models
- running average does not seem to have a really big impact

Not available for evaluation due to unfinished training:
- different architectures
- different mutation parameters

## Outlook
We chose to create our training environment in a way that makes it compatible with reinforcement learning.
[Based on this paper](http://proceedings.mlr.press/v80/lee18b/lee18b.pdf) we planned on implementing a second branch of the ANN predicting a future reward for the predicted current.

## Methods and Remarks
### Alternative Methods
Before deciding on genetic algorithm we discussed different approaches.
Mainly we discussed implementing a road prediction framework and then use this predictions to approximate future states and thus being able to linear solve for an optimal current. However we decided against this, since we would need a really good road prediction in order to achieve good results. But due to a lack of quantity and quality in the given data achieving a good enough road prediction did not seem feasible.
Nevertheless, assuming more data is available one could get a well enough road prediction to feed in as input to the neural networks used in our genetic algorithm in order to enhance their performance.

### Training
We were able to train our model on the grid of the ikw. In order to do this, we had to set up a virtual environment installing all the needed dependencies needed to run our code. Thus we have two .sge files, one setting up the environment and one activating it.

### Organization & Timeline
On the first day of the hackathon we discussed different approaches and decided on one. We then discussed the main structure of our model and set up a Trello Board in which we defined the different steps and goals.
On the second day we implemented our model and planned to train different setups on the grid overnight so we could spend the third day evaluating the different models and tune the parameters. However, due to some difficulties and lack of expertise with the grid this did not work out as planned and we had to spend most of our third day on training.
