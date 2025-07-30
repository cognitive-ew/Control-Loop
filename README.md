# Control-Loop for Cognitive Decision Making
A control loop that drives a cognitive decision maker.
The code is in support of the book "Cognitive EW: An AI Approach" by Karen Zita
Haigh and Julia Andrusenko. 

## Main entry file
**runExperiment.m**<br>
Main entry file. It will run the vanilla versions of the scenario driver and
decision maker. It provides infrastructure for calling a more complete system

## Scenario Driver
**ScenarioDriverVanilla.m**<br>
This code is the test infrastructure around a decision maker. It loads a
ground truth dataset, selects some of that data to be training data, runs a
"real time" test using samples from the original data (which may or may not
have been in the chosen training data). It then evaluates performance as
"Adequacy": the actual performance obtained from executing in the environment
divided by the best known (optimal) performance.<br>

There are several exercises for students:<br>
1. Configuration support for retraining in the decision maker  -- in the SD,
   it's just the boolean of whether retraining is allowed
2. Configuration support for epsilon-greedy exploration in the decision
    maker -- in the SD, it's just the value of epsilon
3. Support multiple ML regression models
4. Evaluate the models by memory and time (in addition to Adequacy)
5. Add n-choose-k ablation trials

The scenario driver corresponds to Algorithm 10.1, Figure 10.13,
and Project 11.6.12 of the second edition, or Figure 10.2 of the first edition.

The following image shows the architectural concept. The scenario driver loads a ground truth dataset, chooses some of the environments that it will use to pretrain the decision maker, and then runs a "real-time" loop. The code uses very simple data replay; the image shows that data replay, augmentation, models, and real-world execution are also appropriate.
![Scenario Driver Architecture](https://github.com/cognitive-ew/Control-Loop/blob/main/images/02%20ScenarioDriver.png)

## Decision Maker
**DecisionMakerVanilla.m**<br>
This code is a simple AI-based decision maker. It supports pretraining an ML
model on an initial dataset, then a "real time" test where it chooses a
decision for a current set of observations. The model is metric = model(
observables, controllables).

There are two student exercises:<br>
1. In mission learning (Update learned model with the feedback from the
  environment)
2. Epsilon-greedy exploration

The decision maker corresponds to Algorithm 5.1 (either edition),
and also Algorithm 10.1 and Project 11.6.9 of the second edition.

In the following image, the decision maker is pretrained on every environment it will encounter. We therefore expect perfect performance.
![Vanilla version of a Decision Maker](https://github.com/cognitive-ew/Control-Loop/blob/main/images/03%20DecisionMaker%20Vanilla.png)

In the following image, the decision maker is pretrained only with E124 and E135. The learner has to update the model when it gets new information from the environment.
![Augmenting the Decision Maker with Reinforcement Learning](https://github.com/cognitive-ew/Control-Loop/blob/main/images/04%20DecisionMaker%20RL.png)


## A Custom Kernel for a Support Vector Regression Machine
**PUK_kernel.m**<br>
A custom kernel for a support vector machine based on the approach of
Ustun, Melssen, and Buydens, "Facilitating the application of Support Vector
Regression by using a Universal Pearson VII Function Based Kernel," Chemometrics 
and Intelligent Laboratory Systems, Vol. 81, No. 1, 2006.
DOI: 10.1016/j.chemolab.2005.09.003.

## Sample Data
**decisionData.csv**<br>
A synthetic dataset computing performance metrics for 8 environments, 3 observables, 
and two controllables. Here are the characteristics of this synthetic data file.

![Architecture of Scenario Driver](https://github.com/cognitive-ew/Control-Loop/blob/main/images/01%20DataDescription.png)

A more realistic dataset would map RF environments (e.g., threats, laydowns) to appropriate responses. For example: (i) free and clear comms, (ii) tone jammer, (iii) blinking jammer, and (iv) co-channel interference. The corresponding techniques could be (i) default operations, (ii) notch filter, (iii) redundant packets, and (iv) directional antenna.

# Evaluating ML Models

## Accuracy, Time, and Memory
Every model has advantages and disadvantages; evaluate a set of models
based on the criteria that matter to your mission. Particularly useful metrics are accuracy, memory, and time. In the on-line learning setting, accuracy of the model will be different at the end of the scenario than at the beginning; we evaluate this improvement using Adequacy. As an example, DeepNets require a lot of training data before they are useful. The following chart shows different models in (a) static and (b) on-line learning settings. We evaluate the static model with normalized root-mean-squared-error, and the on-line learning using Adequacy. (a) nRMSE Accuracy vs Time for a static model trained with an 80/20 train/test split. nRMSE=0.0 is a perfect fit. (b) Adequacy vs Time for a dynamic model that learns in mission starting with k = 4 pretrained environments out of n = 8 total environments. Adequacy=1.0 is optimal.

![Evaluate ML models by accuracy, Adequacy, and time](https://github.com/cognitive-ew/Control-Loop/blob/main/images/06%20Eval%20DecisionMaker.png)

Note that MATLAB's dynamic memory approach means that time and memory usage will not be completely consistent.

## Ablation Tests

Ablation tests determine how much a priori training data is needed to get a desired
accuracy. $n$-choose-$k$ ablation testing proves that a cognitive system is capable of
learning how to handle new environments. An ablation trial evaluates the impact of specific training examples(s) on the generalization ability of the model. The ground
truth data has $n$ known cases; we train the system on $k \subseteq n$ cases, and test on all $n$, for all values of $k$ and all subsets $n$-choose-$k$. Thus, during the test, $n-k$ environments are novel.

![Ablation Tests Determine How Much A Priori Data is required](https://github.com/cognitive-ew/Control-Loop/blob/main/images/05.01%20Ablation%20Intro.png)

In the following image, the decision maker uses a decision tree for reinforcement learning. The $x$-axis is $k$ showing all possible subsets of $n$... that is, when $x=0$, there is no a priori training data, and when $x=n$, the decision maker is trained on all environments it will encounter during the scenario. 
![Ablation Tests for a Decision Tree Model](https://github.com/cognitive-ew/Control-Loop/blob/main/images/05.02%20Ablation%20DecisionTree.png)

In the following image, the decision maker uses a 10 different ML models for reinforcement learning. Support Vector Regression Machines (SVM) with a Pearson VII Universal Kernel obtains approximately 70% Adequacy when compared to optimal... even when pretrained on no training data.
![Ablation Trials with Multiple ML Models](https://github.com/cognitive-ew/Control-Loop/blob/main/images/05.03%20Ablation%20MultipleML.png)
