%%%%%%%%%%
% Scenario Driver (SD)
%
% This code is the test infrastructure around a decision maker. It loads a
% ground truth dataset, selects some of that data to be training data, runs a
% "real time" test using samples from the original data (which may or may not
% have been in the chosen training data). It then evaluates performance as
% "Adequacy": the actual performance obtained from executing in the environment
% divided by the best known (optimal) performance.
%
% There are several student exercises:
% 1. Configuration support for retraining in the decision maker  -- in the SD,
%    it's just the boolean of whether retraining is allowed
% 2. Configuration support for epsilon-greedy exploration in the decision
%    maker -- in the SD, it's just the value of epsilon
% 3. Support multiple ML regression models
% 4. Evaluate the models by memory and time (in addition to Adequacy)
% 5. Add n-choose-k ablation trials
%
% The code is in support of the book "Cognitive EW: An AI Approach" by Karen Zita
% Haigh and Julia Andrusenko. It corresponds to Algorithm 10.1, Figure 10.13,
% and Project 11.6.12 of the second edition, or Figure 10.2 of the first edition.
%
% This work is licensed under a Creative Commons Attribution-NonCommercial 4.0
% International License. You are free to share and adapt the code, provided you
% give appropriate credit. The code is provided as is, without warranty of any
% kind.
%
% (c) 2025 by Karen Zita Haigh, Haskill Consulting LLC, karen@haskillconsulting.com
%
%%%%%%%%%%

classdef ScenarioDriverVanilla
    properties
        debug

        % Data
        datafile
        obsCols
        ctrCols
        metrCols
        X
        y
        strategies
        splitType
        splitStr
        model

        testType           % one of 'first', 'rand'
        testSizes          % for 'first' mode [1, 100, 1000, size(driver.X,1]
        randTrainPercs     % for 'rand' mode, [0.5, 0.99]
        numRandRepeats     % how many times to repeat 'rand' splits
        testStep           % spacing for controller evaluation

        % For the decision maker
        decisionMaker
    end

    methods
        %% Load and preprocess data
        function obj = loadData(obj)
            data = readtable(obj.datafile);
            obj.obsCols = data.Properties.VariableNames(contains(data.Properties.VariableNames, 'obs'));
            obj.ctrCols = data.Properties.VariableNames(contains(data.Properties.VariableNames, 'ctr'));
            obj.metrCols = data.Properties.VariableNames(contains(data.Properties.VariableNames, 'metric'));

            dataX = removevars(data, 'Env');
            obj.X = removevars(dataX, obj.metrCols);
            yTemp = dataX{:, obj.metrCols};
            obj.y = yTemp(:, 1);
        end

        %% Create strategy grid
        function obj = makeStrategies(obj)
            vals = unique(obj.X.(obj.ctrCols{1}));
            strats = vals;
            for i = 2:length(obj.ctrCols)
                newVals = unique(obj.X.(obj.ctrCols{i}));
                strats = combvec(strats', newVals')';
            end
            obj.strategies = strats;
        end

        %% Split data into training
        function [xTrain, yTrain] = splitData(obj)
            if strcmp(obj.splitType{1}, 'rand')
                n = size(obj.X, 1);
                testIdx = randperm(n, round(n * obj.splitType{2}));
                trainIdx = setdiff(1:n, testIdx);
            elseif strcmp(obj.splitType{1}, 'first')
                trainIdx = 1:obj.splitType{2};
            else
                [xTrain, yTrain] = obj.splitData({'rand', 0.2});
                return;
            end
            xTrain = obj.X(trainIdx, :);
            yTrain = obj.y(trainIdx);
        end

        %% Get the actual performance value for this (obs,strat) pair
        %% Here we find best match in available data; this could run a model or a simulation
        function perf = getActualPerformance(obj, obs, strat)
            tol = 1e-5;

            % Sanity check
            % disp(strat);
            assert(isvector(strat) && isnumeric(strat), 'strat must be a numeric vector');

            % Strategy match
            ctrlData = table2array(obj.X(:, obj.ctrCols));  % (N x d_ctr)
            maskStrat = all(abs(ctrlData - strat) < tol, 2);

            % Observation match within strategy-filtered data
            xMatchStrat = obj.X(maskStrat, :);
            obsArray = table2array(obs);  % (1 x d_obs)
            obsData = table2array(xMatchStrat(:, obj.obsCols));  % (M x d_obs)
            maskObs = all(abs(obsData - obsArray) < tol, 2);

            idx = find(maskObs, 1);
            if isempty(idx)
                perf = NaN;
            else
                yStratObs = obj.y(maskStrat);  % grab y values aligned with strat mask
                perf = yStratObs(idx);         % pick performance for matched observation
            end

            %fprintf('Actual for [%s] is %.6f\n', strjoin(string(strat), ', '), perf);
        end

        %% Get the actual performance value for this (obs,strat) pair
        %% Here we find best match in available data; this could run a model or a simulation
        function [stratStr, bestPerf] = getOptimalPerformance(obj, obs)
            tol = 1e-5;
            obsArray = table2array(obs);
            obsFullArray = obj.X{:, obj.obsCols};

            maskObs = all(abs(obsFullArray - obsArray) < tol, 2);
            if ~any(maskObs)
                bestPerf = NaN;
                stratStr = '[NaN]';
                return;
            end

            xMatchObs = obj.X(maskObs, :);
            yMatchObs = obj.y(maskObs);
            [bestPerf, idx] = max(yMatchObs);

            bestCtr = xMatchObs{idx, obj.ctrCols};
            stratStr = ['[', sprintf('%d, ', bestCtr(1:end-1)), sprintf('%d]', bestCtr(end))];
        end

        %% Run controller loop for a model
        function [adequacy, lenTrain, lenTest] = runController(obj, modelFunc)
            [xTrain, yTrain] = obj.splitData();
            lenTrain = size(xTrain,1);
            lenTest = floor(size(obj.X,1) / obj.testStep);

            %fprintf('\n\n=====\n\n')
            %fprintf('Setting obj.decisionMaker with %d training, %d test\n',lenTrain,lenTest)
            obj.decisionMaker = DecisionMakerVanilla(obj);
            obj.decisionMaker = obj.decisionMaker.pretrain(modelFunc, xTrain, yTrain);

            sumAdeq = 0; n = 0;
            for i = 1:obj.testStep:size(obj.X, 1)
                currentObs = obj.X(i, obj.obsCols);
                [~, chosenStrat, predictedPerf] = obj.decisionMaker.decide(currentObs);

                actualPerf = obj.getActualPerformance(currentObs, chosenStrat);
                [optimalStrat, optimalPerf] = obj.getOptimalPerformance(currentObs);
                adeq = actualPerf / optimalPerf;

                sumAdeq = sumAdeq + adeq; n = n + 1;
                if obj.debug
                    fprintf('%4d. Chosen: [%s] A=%.4f P=%.4f, Optimal: %s @ %.4f, Adeq=%.4f\n', ...
                        i, num2str(chosenStrat), actualPerf, predictedPerf, num2str(optimalStrat), optimalPerf, adeq);
                end
            end
            adequacy = sumAdeq / n;
        end

        %% Return models with kernel variants
        function models = getModels(obj)
            models = {};

            % Kernels with untrained model templates
            % models{end+1} = {'SVR_Linear', @(X,y) fitrsvm(X, y, 'KernelFunction', 'linear')};
            models{end+1} = {'SVR_RBF', @(X,y) fitrsvm(X, y, 'KernelFunction', 'rbf', 'KernelScale', 'auto')};
            models{end+1} = {'SVR_VII', @(X, y) fitrsvm(X, y, 'KernelFunction', 'PUK_kernel') };
        end

        %% Evaluate all models on current split
        function runAllModels(obj)
            models = obj.getModels();
            for i = 1:length(models)
                name = models{i}{1};
                modelFunc = models{i}{2};
                [Adeq, lenTrain, lenTest] = obj.runController(modelFunc);
                s1 = sprintf('(%-8d, %-8d)',lenTrain,lenTest);
                fprintf('%-18s %-8.5f (%-25s) %s\n', name, Adeq, obj.splitStr, s1);
            end
        end

        %% Run with train/test mix variations
        function runMixedTests(obj)
            mode = obj.testType;

            s1 = sprintf('(%-8s, %-8s)','lenTrain','lenTest');
            fprintf('%-18s %-8s (%-25s) %s\n', 'ML Model', 'Adeq', 'SplitType:SplitVal', s1);

            if any(strcmp(mode, 'first'))
                for n = obj.testSizes
                    obj.splitType = {'first', n};
                    obj.splitStr = sprintf('First %d', n);
                    obj.runAllModels();
                end
            end

            if any(strcmp(mode, 'rand'))
                for percTest = obj.randTrainPercs
                    for i = 1:obj.numRandRepeats
                        obj.splitType = {'rand', percTest};
                        obj.splitStr = sprintf('Test%%: %.2f', percTest);
                        obj.runAllModels();
                    end
                end
            end
        end
    end
end
