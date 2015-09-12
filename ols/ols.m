clear all; clc
addpath('C:\Users\tmr17\Dropbox\MathWorks File Exchange\normal')
tic
seed =1234;
rng(seed,'twister')

N       = 1e5;
T       = 5;
n       = N*T;

% generate the data
X = [ones(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) 15+3*randn(N*T,1) .7-.1*randn(N*T,1)];

% X coefficients
b      = [ 2.15 0.10  0.50 0.10 .75 1.2 ]';

% std dev of errors
sigAns = 0.3;

% create the observed outcomes
draw = sigAns*randn(N*T,1);
Y=X*b+draw;
% tabulate(Y)
disp(['Time spent generating data: ', num2str(toc),' seconds.'])

% Estimation
bhat = X\Y

options = optimset('Disp','off','LargeScale','on' ,'MaxIter',1e8,'MaxFunEvals',1e8,'TolX',1e-7,'Tolfun',1e-7,'DerivativeCheck','off','GradObj','on' ,'FinDiffType','central');
startval = [.5*ones(size(bhat));.5];
% [bEst,~,~,~,~,hEst] = fminunc('normalMLE',startval,options,[],Y,X,ones(size(Y)),ones(size(Y)));
bEst = fminunc('normalMLE',startval,options,[],Y,X,ones(size(Y)),ones(size(Y)));
% [bEst sqrt(diag(hEst\eye(size(hEst))))]
bEst

toc