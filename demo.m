%% Demo for SCD-SOMP algorithm
% The current fuction 'SCD_TrainSOMP' selects samples with maximum SOMP residue for each iteration to learn trace/target materials. Change as necessary based on your desired fitting error objective.

function dictionary = demo

% demo with indian pines scene
load('/data/IndianPines.mat');

% Run the dictionary, and pre-process as needed like band selection and de-noising algorithms
% demo run with 30 atoms and 8,000 iterations. Change as necessary.
dictionary = SCD_TrainSOMP(IP, 30, 8000);
clc;
save('/results/dictionary', 'dictionary');

end