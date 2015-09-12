function [like,grad]=clogitAdj1(b,restrMat,Y,X,Z,baseAlt,W,adj,beta)
%CLOGITADJ estimates a conditional logit model
%   LIKE = CLOGITADJ(B,RESTRMAT,Y,X,Z,BASEALT,W,ADJ,BETA) 
%   estimates McFadden's choice model, which is a special case of the 
%   conditional logistic regression model. There are J choice
%   alternatives. Parameter restrictions are constructed in RESTRMAT.
%   
%   For estimation without restrictions, set RESTRMAT to be an empty matrix. 
%   
%   Y is an N x 1 vector of integers 1 through J indicating which
%   alternative was chosen. 
%   X is an N x K1 matrix of individual-specific covariates.
%   Z is an N x K2 x J array of covariates that are alternative-specific.
%   BASEALT is the integer number of the category in Y that will be used
%   as the reference alternative. Alternative J is the default.
%   W is an N x 1 vector of weights
%   ADJ is an N X J matrix of adjustment terms, generally FV terms
%   BETA is a scaling variable for ADJ, generally a discount factor
%   B is the parameter vector, with (J-1)*K1 + K2 elements
%   
%   CLOGITADJ can estimate one of three possible models:
%   1. Multinomial logit model: Z is empty
%   2. Conditional logit model: X is empty
%   3. Alternative-specific conditional logit model: X and Z both non-empty
%   
%   This function does *not* automatically include a column of ones in X.
%   It also does *not* automatically drop NaNs
%   
%   Parameters are ordered as follows: {X parameters for alternative 1,
%   ..., X parameters for alternative J, Z parameters}
%   
%   Reference: McFadden, D. L. 1974. "Conditional Logit Analysis of
%   Qualitative Choice Behavior." in Frontiers in Econometrics, ed.
%   P. Zarembka, 105–142. New York: Academic Press.

% Copyright 2014 Jared Ashworth and Tyler Ransom, Duke University
% Special thanks to Vladi Slanchev and StataCorp's asclogit command
% Revision History: 
%   August 15, 2014
%     Created 
%   August 20, 2014
%     Changed name of function to match filename; added error checks
%==========================================================================

% error checking
assert((~isempty(X) || ~isempty(Z)) && ~isempty(Y),'You must supply data to the model');

N  = size(Y,1);
K1 = size(X,2);
K2 = size(Z,2);
J  = numel(unique(Y));

assert(length(b)==(K1*(J-1)+K2)   ,'b has the wrong number of elements');
assert(ndims(b)==2 && size(b,2)==1,'b must be a column vector');
assert(ndims(Y)==2 && size(Y,2)==1,'Y must be a column vector');
assert(  min(Y)==1 && max(Y)==J   ,'Y should contain integers numbered consecutively from 1 through J');
if ~isempty(X)
	assert(ndims(X)==2  ,'X must be a 2-dimensional matrix');
	assert(size(X,1)==N ,'The 1st dimension of X should equal the number of observations in Y');
end
if ~isempty(Z)
	assert(ndims(Z)==3  ,'Z must be a 3-dimensional array');
	assert(size(Z,1)==N ,'The 1st dimension of Z should equal the number of observations in Y');
	assert(size(Z,3)==J ,'The 3rd dimension of Z should equal the number of alternatives in Y');
end
if nargin>=7 && ~isempty(W)
	assert(ndims(W)==2 && size(W,2)==1 && length(W)==N,'W must be a column vector the same size as Y');
end
if nargin>=8 && ~isempty(adj)
	assert(size(adj,2)==J,'The 2nd dimension of adj should equal the number of alternatives in Y');
end

% coef vector for Z variables
b2    = b(K1*(J-1)+1:K1*(J-1)+K2);

% apply restrictions as defined in restrMat
if ~isempty(restrMat)
	b = applyRestr(restrMat,b);
end

% default values if not included
if nargin<6 || isempty(baseAlt)
	baseAlt = J;
end

if nargin<7 || isempty(W)
	W = ones(N,1);
end

if nargin<8 || isempty(adj)
	adj=zeros(N,J);
end

if nargin<9 || isempty(beta)
	beta=0.9;
end

% initialize values
num   = zeros(N,1);
num1  = zeros(N,1);
dem   = zeros(N,1);
dem1  = zeros(N,1);
numer = ones(N,J);
numer1= ones(N,J);

if K1==0 && K2>0
	% sets BASEALT to be the alternative that is normalized to zero
	flagger = setdiff(1:K2,baseAlt);
	for j=1:J
		temp = (Z(:,flagger,j)-Z(:,flagger,baseAlt))*b2(flagger) + beta*(adj(:,j)-adj(:,baseAlt));
		num  = (Y==j).*temp+num;
		dem  = exp(temp)+dem;
		numer(:,j) = exp(temp);
	end
	P=numer./(dem*ones(1,J));
	
	like=-W'*(num-log(dem));
	
	% analytical gradient
	numg = zeros(K2,1);
	demg = zeros(K2,1);
	
	grad = zeros(size(b));
	for j=1:J
		numg(flagger) = -(Z(:,flagger,j)-Z(:,flagger,baseAlt))'*(W.*(Y==j))+numg(flagger);
		demg(flagger) = -(Z(:,flagger,j)-Z(:,flagger,baseAlt))'*(W.*P(:,j))+demg(flagger);
	end
	grad(flagger)=numg(flagger)-demg(flagger);
end