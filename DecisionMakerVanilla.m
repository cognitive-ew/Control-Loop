%%%%%%%%%%
% Decision Maker
%
% This code is a simple AI-based decision maker. It supports pretraining an ML
% model on an initial dataset, then a "real time" test where it chooses a
% decision for a current set of observations. The model is metric = model(
% observables, controllables).
% 
% There are two student exercises:
% - In mission learning (Update learned model with the feedback from the
%   environment)
% - Epsilon-greedy exploration
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

classdef DecisionMakerVanilla
    properties
        strategies       % Matrix of possible strategies to evaluate
        obsCols          % Column names for observation features
        ctrCols          % Column names for control features
        xCols            % Combined obs+ctr names used for training
        trainedModel     % Model trained to predict strategy performance

%% Student exercise -- Retraining
   %% Student exercise -- Exploration
    end

    methods
        % Constructor initializes strategy set and feature columns
        function obj = DecisionMakerVanilla(experiment)
            obj.strategies = experiment.strategies;
            obj.xCols = [experiment.obsCols, experiment.ctrCols];
            obj.obsCols = experiment.obsCols;
            obj.ctrCols = experiment.ctrCols;

%% Exercise: Prep for exploration
%% Exercise: Prep for retraining
        end

        % Initial training of the model using provided data
        function obj = pretrain(obj, modelFunc, X, y)
            obj.trainedModel = modelFunc(X, y);

%% Exercise: Prep for retraning.
%% Store untrained base model
%% Store the initial data (new observations will append)
        end

        % Decision function: 
        % Selects the best strategy based on current observation
        % Returns updated object, selected strategy, and predicted score
        % Exercise: Add the previous observation, the previous chosen strategy, and the actual performance seen after execution in the environment to the dataset.
        function [obj, bestStrat, bestScore] = decide(obj, currObs, ~)
            obsArray = table2array(currObs);
            bestScore = -inf;
            bestStrat = obj.strategies(1,:);

%% Exercise: Retrain the model with newest data
%% Exercise: With probability epsilon, choose a random strategy to encourage exploration

            % Greedy selection
            % Evaluate all strategies and choose the one with the highest predicted score
            for i = 1:size(obj.strategies, 1)
                strat = obj.strategies(i, :);
                inputTbl = array2table([obsArray, strat], 'VariableNames', obj.xCols);
                try
                    preds = predict(obj.trainedModel, inputTbl);
                    if preds(1) > bestScore
                        bestScore = preds(1);
                        bestStrat = strat;
                    end
                catch
                    % Silently ignore prediction errors
                end
            end
        end
    end
end


%classdef DecisionMakerVanilla
%    properties
%        strategies
%        obsCols
%        ctrCols
%        xCols
%        trainedModel
%    end
%
%    methods
%        function obj = DecisionMakerVanilla(experiment)
%            obj.strategies = experiment.strategies;
%            obj.xCols = [experiment.obsCols, experiment.ctrCols];
%            obj.obsCols = experiment.obsCols;
%            obj.ctrCols = experiment.ctrCols;
%        end
%
%        function obj = train(obj, modelFunc, X, y)
%            try
%                obj.trainedModel = modelFunc(X, y);
%                %fprintf('Trained on %d\n',size(X,1));
%            catch ME
%                warning('Model training failed: %s',ME.message);
%                rethrow(ME);
%            end
%        end
%
%        function [bestStrat, bestScore] = decide(obj, currObs)
%            obsArray = table2array(currObs);
%            bestScore = -inf;
%            bestStrat = obj.strategies(1,:);
%            %fprintf('Initial bestScore is %s, bestStrat is %s\n',class(bestScore),class(bestStrat))
%            %fprintf('Initial BestStrat=%s, bestScore=%.4f\n',num2str(bestStrat),bestScore)
%
%            for i = 1:size(obj.strategies, 1)
%                strat = obj.strategies(i, :);
%                instance = [obsArray, strat];
%                inputTbl = array2table(instance, 'VariableNames', obj.xCols);
%                try
%                    preds = predict(obj.trainedModel, inputTbl);
%                    %fprintf('Predicted: %.4f\n',preds(1))
%                    if preds(1) > bestScore
%                        bestScore = preds(1);
%                        bestStrat = strat;
%                    end
%                catch ME
%                    warning('Decide failed: %s',ME.message)
%                    rethrow(ME);
%                end
%            end
%            %fprintf('Final BestStrat=%s, bestScore=%.4f\n',num2str(bestStrat),bestScore)
%        end
%    end
%end
