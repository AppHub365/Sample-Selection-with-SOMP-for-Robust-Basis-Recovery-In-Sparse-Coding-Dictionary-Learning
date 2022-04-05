%% SCD (Sparse Coding Dictionary Learning) with SOMP sampling
% Published paper (IEEE LOCS): https://doi.org/10.1109/LOCS.2019.2938446
% SCD source code originally authored by Adam Charles and this code is
% modified by Ayan Chatterjee. Email: ayan@outlook.com
% This code supports compatible GPU (optional). Please see relevant links.

%% Relevant Links (Last accessed 10 September 2019)
% Adam's SCD Paper (IEEE JSTP): https://doi.org/10.1109/JSTSP.2011.2149497
% Adam's SCD code: http://adamsc.mycpanel.princeton.edu/documents/Dictionary_Learning_Library_v1-0.zip
% SD-SOMP paper (IEEE JSTP): https://doi.org/10.1109/JSTSP.2015.2410763 
% SD-SOMP source code: https://mx.nthu.edu.tw/~tsunghan/download/Demo_LqSDSOMP.rar

%% Inputs
% Image -> 3D hyperspectral or multispectral cube
% num_atoms -> number of atoms to learn
% maxIter -> maximum number of iterations

function dictionary = SCD_TrainSOMP(Image, num_atoms, maxIter)

clc;
[h, w, nB] = size(Image);
Image = reshape(single(Image), [h*w, nB]);

opts.save_name = 'temp_1.mat';      % Save Name in case of errors
opts.sparse_type = 'l1ls_nneg';     % Choose to use l1ls for sparsification OMP_qr
opts.grad_type = 'norm';            % Choose weather to include the Forb norm in E(a,D)
opts.n_elem = num_atoms;            % Number of dictionary elements
opts.iters = maxIter;               % Number of learning iterations
opts.in_iter = 200;                 % Number of internal iterations
opts.GD_iters = 1;                  % Basis Gradient Descent iterations
opts.step_size = 0.01;              % Initial Step Size for Gradient Descent
opts.decay = 0.9998;                % Step size decay factor
opts.lambda = 0.5;                  % Lambda Value for Sparsity
opts.tol = 0.001;                   % Sparsification Tolerance
opts.verb = 1;                      % Default to no verbose output
opts.ssim_flag = 0;                 % Default to no normalization between samples
opts.std_min = 0.1;                 % Default to min. sample standard deviation of 0.1

dictionary_initial = abs(rand(nB, opts.n_elem)); %create a random initial dictionary
dictionary_initial = dictionary_initial./(ones(nB, 1)*sqrt(sum(dictionary_initial.^2, 1))); % Basis normalized for l2 norm
dictionary = learn_dictionary(Image', dictionary_initial, @l1ls_nneg_wrapper, opts); %learn dictionary
dictionary = dictionary';

end

function [dictionary_end] = learn_dictionary(data_obj, initial_dict, infer_handle, opts)
% OPTIONS: Make sure that the correct options are set and that all
% necessary variables are available or defaulted.
if ~isfield(opts, 'ssim_flag')
    opts.ssim_flag = 0; % Default to no normalization between samples
end

if (~isfield(opts, 'std_min'))
    opts.std_min = 0.1; % Default to min. sample standard deviation of 0.1
    warning('Inputs:UnspecifiedParam',  ...
        ['Min sample STD not set by user!! Using STDmin = 0.1. ', ... 
        'This is probably bad! ', ...
        'Ctrl-C and restarting with a specified value is recommended.'])
end

if ~isfield(opts, 'save_name')
    date_str = date;
    opts.save_name = [date_str(8:end), date_str(3:7), date_str(1:2), ...
        'Dictionary_' num2str(opts.n_elem), 'Elems_', num2str(opts.lambda), ...
        'lambda.mat'];
    fprintf('Save name not specified, saving as %s...\n', opts.save_name)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Error Checking
if opts.n_elem ~= size(initial_dict, 2)
    error('Dimension mismatch between opts.n_elem and initial dictionary size!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations and Dimention Extraction
% Initialize Basis
dictionary_n = initial_dict; 
% Iteration counter initialization
iter_num = 0;
% Initialize step size
step_s = opts.step_size;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run Algorithm
fprintf('Educating your basis...\n')
basic_cell.options = opts;
while iter_num < opts.iters
    try
        %% Get Training Data.
        if(gpuDeviceCount >= 1)
            SOMP_residue = estSOMPresidue(gpuArray(dictionary_n), gpuArray(data_obj));
            SOMP_residue = gather(SOMP_residue);
        else
            SOMP_residue = estSOMPresidue(dictionary_n, data_obj);
        end
        [~, data_use_ind] = sort(SOMP_residue, 'descend'); % descending order of SOMP
        data_use_ind = data_use_ind(1:opts.in_iter);
        x_im = data_obj(:, data_use_ind); % Initialize matricies that will be populated during the actual learning
        %% Interior loop: find sparse coefficients 
        coef_vals = gen_multi_infer(dictionary_n, x_im, infer_handle, opts);
        %% Minimize the energy w.r.t. the dictionary using gradient descent
       	dictionary_n = dictionary_update(x_im, dictionary_n, coef_vals, step_s, opts);
        iter_num = iter_num + 1; % update the iteration count
        if opts.verb == 1
            %Spit out info
            im_snr = mean(sum(x_im.^2, 1)./sum((x_im - dictionary_n*coef_vals).^2, 1));
            disp(strcat("Iter: ", num2str(iter_num),", SNR: ", num2str(im_snr),", mean SOMP: ", num2str(mean(SOMP_residue)),", step size is ", num2str(step_s)));
        end
        % Update the step size
        step_s = step_s*opts.decay;
    catch ME
        fprintf('Saving last dictionary before error...\n')
        basic_cell.dictionary = dictionary_n;
        basic_cell.iter = iter_num;
        eval(sprintf('save %s basic_cell;', opts.save_name));
        fprintf(ME.message)
        fprintf('The program failed. Your dictionary at the last iteration was saved.')
        rethrow(ME)
    end 
end
dictionary_end = dictionary_n;

end

% Function added by Ayan
function SOMP_residue = estSOMPresidue(dictionary_n, data_obj)
P = eye(size(dictionary_n, 1)) - dictionary_n*pinv(dictionary_n);
SOMP_residue = sum(abs(P*data_obj).^2, 1).^0.5; % l-2 SOMP
end

function coef_vals = gen_multi_infer(dictionary_n, x_im, infer_hand, opts)
% Initialize coefficients
coef_vals = zeros(opts.n_elem, opts.in_iter);
%% Perform the L1-regulized optimization on the data
for index_in = 1:opts.in_iter
    coef_vals(:, index_in) = feval(infer_hand, dictionary_n, x_im(:,index_in), opts);
end

end

function coef_vals = l1ls_nneg_wrapper(dictionary_n, x_im, opts)
coef_vals = l1_ls_nonneg(dictionary_n, x_im, opts.lambda, opts.tol, 1);
end

function x = l1_ls_nonneg(dictionary, varargin)
% IPM PARAMETERS
MU              = 2;        % updating parameter of t
MAX_NT_ITER     = 400;      % maximum IPM (Newton) iteration

% LINE SEARCH PARAMETERS
ALPHA           = 0.01;     % minimum fraction of decrease in the objective
BETA            = 0.5;      % stepsize decrease factor
MAX_LS_ITER     = 100;      % maximum backtracking line search iteration

if (nargin >= 3)
    [~, nBands] = size(dictionary);
    y  = varargin{1};
    lambda = varargin{2};
    varargin = varargin(3:end);
else
    x = [];
    return;
end

% VARIABLE ARGUMENT HANDLING
t0         = min(max(1,1/lambda),nBands/1e-3);
defaults   = {1e-3,false,1e-3,5000,ones(nBands,1),t0};
given_args = ~cellfun('isempty',varargin);
defaults(given_args) = varargin(given_args);
[reltol,~,eta,pcgmaxi,x,t] = deal(defaults{:});

f = -x;

% RESULT/HISTORY VARIABLES
pobjs = [] ; dobjs = [] ; sts = [] ; pitrs = []; pflgs = [];
dobj  =-Inf;
s   = Inf;
conv_iter_n  = 0;
conv_flag  = 0 ;

ntiter  = 0;
lsiter  = 0;
conv_vector =  zeros(nBands, 1);

% diagxtx = diag(At*A);
diagxtx = 2*ones(nBands,1);
%------------------------------------------------------------
%               MAIN LOOP
%------------------------------------------------------------

for ntiter = 0:MAX_NT_ITER
    
    z = dictionary*x-y;
    
    %------------------------------------------------------------
    %       CALCULATE DUALITY GAP
    %------------------------------------------------------------

    nu = 2*z;

    minAnu = min(dictionary'*nu);
    if (minAnu < -lambda)
        nu = nu*lambda/(-minAnu);
    end
    pobj  =  z'*z+lambda*sum(x,1);
    dobj  =  max(-0.25*nu'*nu-nu'*y,dobj);
    gap   =  pobj - dobj;

    pobjs = [pobjs pobj];
    dobjs = [dobjs dobj];
    sts = [sts s];
    pflgs = [pflgs conv_flag];
    pitrs = [pitrs conv_iter_n];

    %------------------------------------------------------------
    %   STOPPING CRITERION
    %------------------------------------------------------------

    if (gap/abs(dobj) < reltol)
        return; %if solved
    end
    %------------------------------------------------------------
    %       UPDATE t
    %------------------------------------------------------------
    if (s >= 0.5)
        t = max(min(nBands*MU/gap, MU*t), t);
    end

    %------------------------------------------------------------
    %       CALCULATE NEWTON STEP
    %------------------------------------------------------------
    
    d1 = (1/t)./(x.^2);

    % calculate gradient
    gradphi = dictionary' * (z*2) + lambda-(1/t)./x;
    
    % calculate vectors to be used in the preconditioner
    prb     = diagxtx + d1;

    % set pcg tolerance (relative)
    normg   = norm(gradphi);
    pcgtol  = min(1e-1,eta*gap/min(1,normg));
    
    if (ntiter ~= 0 && conv_iter_n == 0)
        pcgtol = pcgtol*0.1;
    end
    
    [conv_vector, conv_flag, ~, conv_iter_n,~] =...
        pcg(@AXfunc_l1_ls, -gradphi, pcgtol, pcgmaxi,@Mfunc_l1_ls,[],...
        conv_vector, dictionary, d1, 1./prb); %the 4 inputs into both functions

    if (conv_flag == 1)
        conv_iter_n = pcgmaxi;
    end
    
    %------------------------------------------------------------
    %   BACKTRACKING LINE SEARCH
    %------------------------------------------------------------
    phi = z'*z+lambda*sum(x)-sum(log(-f))/t;
    s = 1.0;
    gdx = gradphi'*conv_vector;
    for lsiter = 1:MAX_LS_ITER
        newx = x+s*conv_vector;
        newf = -newx;
        if (max(newf) < 0)
            newz   =  dictionary*newx-y;
            newphi =  newz'*newz+lambda*sum(newx)-sum(log(-newf))/t;
            if (newphi-phi <= ALPHA*s*gdx)
                break;
            end
        end
        s = BETA*s;
    end
    if (lsiter == MAX_LS_ITER)
        break;
    end % exit by BLS
        
    x = newx;
    f = newf;
end

%------------------------------------------------------------
%       ABNORMAL TERMINATION (FALL THROUGH)
%------------------------------------------------------------
return;

end

%------------------------------------------------------------
%       COMPUTE AX (PCG)
%------------------------------------------------------------
function y = AXfunc_l1_ls(conv_vector, dict, d1, ~)
y = (dict'*((dict*conv_vector)*2)) + d1.*conv_vector;
end

%------------------------------------------------------------
%       COMPUTE P^{-1}X (PCG)
%------------------------------------------------------------
function y = Mfunc_l1_ls(conv_vector, ~, ~, prb_inv)
y = prb_inv.*conv_vector;
end

function dict_new = dictionary_update(x_im, dict_old, coef_vals, step_s, opts)

% function dict_new = dictionary_update(x_im, dictionary_old, coef_vals,
% step_s, opts)
% 
% Takes a gradient step with respect to the sparsity inducing energy
% function.
% 
% Inputs:
%   x_im        - Data samples over which to average the gradient step
%   dict_old    - The previous dictionary (used to infer the coefficients)
%   coef_vals   - The inferred coefficients for x_im using dict_old
%   step_s      - The step size to take in the gradient direction
%   opts        - Options for the particular problem (outlined in
%                 learn_dictionary.m)
%
% Outputs:
%   dict_new    - The new dictionary after the gradient step
% 
% Last Modified 6/4/2010 - Adam Charles

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Take a gradient step
if strcmp(opts.grad_type, 'norm')
    for index2 = 1:opts.GD_iters
        % Take a step in the negative gradient of the basis:
        % Minimizing the energy:
        % E = ||x-Da||_2^2 + lambda*||a||_1^2
        % Update The basis matrix
        updateTerm = (x_im - dict_old*coef_vals)*coef_vals';
        dict_new = dict_old + step_s*updateTerm;
        dict_new = dict_new*diag(1./(sqrt(sum(dict_new.^2)))); % Re-normalize the basis for l2 norm
    end     
elseif strcmp(opts.grad_type, 'forb')
    for index2 = 1:opts.GD_iters
        % Take a step in the negative gradient of the basis:
        % This time the Forbenious norm is used to reduce unused
        % basis elements. The energy function being minimized is
        % then:
        % E = ||x-Da||_2^2 + lambda*||a||_1^2 + ||D||_F^2

        % Update The basis matrix
        dict_new = dict_old + (step_s)*((x_im - dict_old*coef_vals)*coef_vals'...
            - opts.lambda2*2*dict_old)*diag(1./(1+sum(coef_vals ~= 0, 2)));
    end  
end
end