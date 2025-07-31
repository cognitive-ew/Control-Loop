%%%%%%%%%%
% runExperiment.m
%
% Main entry point for control loop and cognitive decision maker.  Scaffolding
% in place for a complete solution that would contain the solutions to
% student exercises.
%
% The code is in support of the book "Cognitive EW: An AI Approach" by Karen Zita
% Haigh and Julia Andrusenko. It corresponds to Algorithm 5.1 (either edition),
% and also Algorithm 10.1 and Project 11.6.9 of the second edition.
% 
% This work is licensed under a Creative Commons Attribution-NonCommercial 4.0
% International License. You are free to share and adapt the code, provided you
% give appropriate credit. The code is provided as is, without warranty of any
% kind.
%
% (c) 2025 by Karen Zita Haigh, Haskill Consulting LLC, karen@haskillconsulting.com
%
%%%%%%%%%%
					
clear; clc;

vanilla = true;
doAblation = false;

if vanilla
    driver = ScenarioDriverVanilla();
%else
%    driver = ScenarioDriverComplete();
end
driver.datafile = 'decisionData.csv';
driver.debug = false;

driver = driver.loadData();
driver = driver.makeStrategies();

driver.testStep = 10; % Generate a test observation every # instances
driver.testSizes = [100, 2000, size(driver.X,1)];
driver.randTrainPercs = [0.5];%, 0.99];
driver.numRandRepeats = 1;

if vanilla
    driver.testType = 'first';
    driver.runMixedTests();

    driver.testType = 'rand';
    driver.runMixedTests();

    %fprintf('\n=====\n');
    %driver.testType = {'rand','first'};
    %driver.runMixedTests()

% else
%     % Retraining and experimentation
%     for tT = {'rand'} %{'first', 'rand'}
%         driver.testType = tT;
%         for eR = [false]%, true]
%             for eE = [0.0]%, 0.05, 0.1]
%                 fprintf('enableRetrain = %d, exploreEpsilon = %.2f\n',eR,eE)
% 
%                 driver.enableRetrain = eR;
%                 driver.exploreEpsilon = eE;
%                 driver.runMixedTests();
%             end
%         end
%     end
% 
%     if doAblation
%         % Ablation
%         driver.testType = 'ablation';
% 
%         driver.enableRetrain = false;
%         driver.exploreEpsilon = 0.0;
%         driver.ablationRange = 2:3; % 0:numel(unique(driver.env));
%         driver.runMixedTests();
% 
%         driver.enableRetrain = true;
%         driver.exploreEpsilon = 0.0;
%         driver.ablationRange = 2:3; % 0:numel(unique(driver.env));
%         driver.runMixedTests();
% 
%         % With retraining and exploration
%         driver.enableRetrain = true;
%         driver.exploreEpsilon = 0.1;
%         driver.runMixedTests();
%     end
end
