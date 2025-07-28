% runExperiment.m
clear; clc;

vanilla = false;
doAblation = false;

if vanilla
    driver = ScenarioDriverVanilla();
else
    driver = ScenarioDriverComplete();
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

else
    % Retraining and experimentation
    for tT = {'rand'} %{'first', 'rand'}
        driver.testType = tT;
        for eR = [false]%, true]
            for eE = [0.0]%, 0.05, 0.1]
                fprintf('enableRetrain = %d, exploreEpsilon = %.2f\n',eR,eE)

                driver.enableRetrain = eR;
                driver.exploreEpsilon = eE;
                driver.runMixedTests();
            end
        end
    end

    if doAblation
        % Ablation
        driver.testType = 'ablation';

        driver.enableRetrain = false;
        driver.exploreEpsilon = 0.0;
        driver.ablationRange = 2:3; % 0:numel(unique(driver.env));
        driver.runMixedTests();

        driver.enableRetrain = true;
        driver.exploreEpsilon = 0.0;
        driver.ablationRange = 2:3; % 0:numel(unique(driver.env));
        driver.runMixedTests();

        % With retraining and exploration
        driver.enableRetrain = true;
        driver.exploreEpsilon = 0.1;
        driver.runMixedTests();
    end
end