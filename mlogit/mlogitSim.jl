## Simple simulation of mlogit
#
# Tyler Ransom
# Duke, June 25, 2015

using Logging
using NLopt
using Optim
using Debug
using HDF5, JLD

# f = open("mlogitSimHess.log","w")
# Logging.configure(filename="mlogitSimHess.log")
# f = redirect_stdout()

function applyRestrGrad(restrMat,grad)
	#APPLYRESTRGRAD applies restrictions to the gradient of an objective function
	#   G = APPLYRESTR(RESTRMAT,GRAD) implements restrictions on the gradient
	#   vector GRAD of parameters according to the specifications found in
	#   RESTRMAT. See APPLYRESTR for more details on constructing the
	#   restriction matrix RESTRMAT.

	# Copyright 2014 Jared Ashworth and Tyler Ransom, Duke University
	# Special thanks to Vladi Slanchev
	# Revision History: 
	#   July 3, 2013
	#     Created
	#   July 9, 2014
	#     Generalize gradient to fit any objective function
	#   July 19, 2014
	#     Published
	#----------------------------------------------------------------------------
	restrMat = restrMat[restrMat[:,1]>0,:] # Remove empty rows

	gRestr=grad
	restrMat=sortrows(restrMat) # need to specify the column that the sorting is based on ???
	R = size(restrMat,1)
	if R>0
		# gradient
		for r=1:R
			i = restrMat[r,1]
			h = restrMat[r,2]
			gRestr[i]=0
			if restrMat[r,3]==1
				gRestr[h]=gRestr[h]+restrMat[r,4]*grad[i]
			end
		end
	end	

	return gRestr
end

function applyRestr(restrMat,b,H)
#APPLYRESTR applies restrictions to a model
#   B = APPLYRESTR(RESTRMAT,B) implements restrictions on the vector B 
#   of parameters according to the specifications found in RESTRMAT. 
#   Two types of restrictions are supported:
#
#     Type 1  Restricting one parameter ("parmA") to equal a fixed value
#     Type 2  Restricting one parameter, parmA, to equal another ("parmB"),
#             potentially multiplied by some real number q and addd to
#             some constant m, e.g. parmA = m + q*parmB.
#   
#   RESTRMAT follows a very specific format. It is an R-by-5 matrix, 
#   where R is the number of restrictions. The role of each of the four 
#   columns is as follows
#
# 	  Column 1  The index of parmA
# 	  Column 2  The index of parmB (zero if type 1 restriction)
# 	  Column 3  Binary vector where 0 indciates a type 1 restriction (parmA
#               set equal to fixed value) and 1 indicates a type 2 
#               restriction (parmA set equal to parmB)
# 	  Column 4  If a type 1 restriction, 0. If a type 2 restriction, any 
#               real number q such that parmA = q*parmB.
# 	  Column 5  If a type 1 restriction, the fixed value. If a type 2
#               restriction, any real number m such that parmA = m+q*parmB.
#
#   APPLYRESTR does not allow for any combination of restrictions. If 
#   two parameters are to be restricted to the same fixed value, they
#   should both be type 1 restrictions rather than a type 1 restriction
#   and a type 2 restriction
#   
#   The same parameter cannot appear in Column 1 of RESTRMAT twice. For 
#   restrictions involving multiple parameters, e.g. b(1) = b(2) = b(3),
#   create two restrictions: 1) b(1) = b(3); and 2) b(2) = b(3).
#
#   Note that RESTRMAT must be sorted in ascending order based on 
#   column 1. This is especially important for hessian maatrix
#   manipulation, explained below.
#
#   [B,INVH] = APPLYRESTR(RESTRMAT,B,H) takes as input a hessian matrix H
#   (typically from an optimization routine) and returns an inverted
#   hessian INVH where restrictions have been applied. A type 1 restriction
#   results in a row and a column of zeroes in the hessian at the index for
#   that paramter. A type 2 restriction duplicates
#   the rows and columns for parmA and parmB. This implies that the 
#   covariance between parmA and parmB is set equal to their variance.
#   Moreover, the covariance of parmA and parmB with other parameters is
#   restricted to be equal.
#
# 
# Copyright 2014 Jared Ashworth and Tyler Ransom, Duke University
# Special thanks to Vladi Slanchev
# Revision History: 
#   July 3, 2013
#     Created
#   November 13, 2013
#     Error message if restrMat is empty or wrong size
#   November 15, 2013
#     Remove empty rows
#   July 19, 2014
#     Published
#----------------------------------------------------------------------------
# assert(~isempty(restrMat),"Empty restriction matrix");
# assert(size(restrMat,2)==5,"Restriction matrix requires 5 columns");

restrMat = restrMat[restrMat[:,1]>0,:]; # Remove empty rows
# assert(~isempty(restrMat),"Restriction matrix has no positive indices");
# assert(maximum(restrMat[:,1])<=size(b,1),"Restriction matrix has indices beyond size of parameter vector");


bRestr=b;
restrMat=sortrows(restrMat,1);
R = size(restrMat,1);
for r=1:R
	if restrMat[r,3]==0
		bRestr[restrMat[r,1]]=restrMat[r,5];
	elseif restrMat[r,3]==1
		bRestr[restrMat[r,1]]=restrMat[r,5]+restrMat[r,4]*bRestr[restrMat[r,2]];
	end
end

if nargin==3
	for r=R:-1:1
		H[restrMat[r,1],:]=[];
		H[:,restrMat[r,1]]=[];
	end
	invH = full(H)\eye(size(H));
    for r=1:R
		invH= [ invH[1:restrMat[r,1]-1,:]; zeros[1,size[invH,1]]; invH[restrMat[r,1]:end,:]];
		invH= [ invH[:,1:restrMat[r,1]-1]  zeros[size[invH,1],1]  invH[:,restrMat[r,1]:end]];
    end
    for r=1:R
        if restrMat[r,3]==1
			invH[restrMat[r,1],:]=restrMat[r,4]*invH[restrMat[r,2],:];
			invH[:,restrMat[r,1]]=restrMat[r,4]*invH[:,restrMat[r,2]];
        end
    end
    invHrestr = invH;
end
return bRestr,invHrestr

end

# function clogit(b::Vector{Float64},restrMat::Matrix{Float64},Y::Vector{Float64},X::Matrix{Float64},Z::Array{Float64,3},baseAlt::Int64=J,W::Vector{Float64}=ones(size(Y,1),1))
function clogit(b,restrMat,Y,X,Z,baseAlt=J,W=ones(size(Y,1),1))
	#CLOGIT estimates a conditional logit model
	#   LIKE = CLOGIT(B,RESTRMAT,Y,X,Z,BASEALT,W) 
	#   estimates McFadden's choice model, which is a special case of the 
	#   conditional logistic regression model. There are J choice
	#   alternatives. Parameter restrictions are constructed in RESTRMAT.
	#   
	#   For estimation without restrictions, set RESTRMAT to be an empty matrix. 
	#   
	#   Y is an N x 1 vector of integers 1 through J indicating which
	#   alternative was chosen. 
	#   X is an N x K1 matrix of individual-specific covariates.
	#   Z is an N x K2 x J array of covariates that are alternative-specific.
	#   BASEALT is the integer number of the category in Y that will be used
	#   as the reference alternative. Alternative J is the default.
	#   W is an N x 1 vector of weights
	#   B is the parameter vector, with (J-1)*K1 + K2 elements
	#   
	#   CLOGIT can estimate one of three possible models:
	#   1. Multinomial logit model: Z is empty
	#   2. Conditional logit model: X is empty
	#   3. Alternative-specific conditional logit model: X and Z both non-empty
	#   
	#   This function does *not* automatically include a column of ones in X.
	#   It also does *not* automatically drop NaNs
	#   
	#   Parameters are ordered as follows: {X parameters for alternative 1,
	#   ..., X parameters for alternative J, Z parameters}
	#   
	#   Reference: McFadden, D. L. 1974. "Conditional Logit Analysis of
	#   Qualitative Choice Behavior." in Frontiers in Econometrics, ed.
	#   P. Zarembka, 105–142. New York: Academic Press.

	# Copyright 2014 Jared Ashworth and Tyler Ransom, Duke University
	# Special thanks to Vladi Slanchev and StataCorp's asclogit command
	# Revision History: 
	#   July 28, 2014
	#     Created
	#   August 15, 2014
	#     Added weights improved readability
	#   August 20, 2014
	#     Added error checks
	#--------------------------------------------------------------------------

	# error checking
	# assert((~isempty(X) || ~isempty(Z)) && ~isempty(Y),"You must supply data to the model")

	N  = size(Y,1)
	K1 = size(X,2)
	K2 = size(Z,2)
	J  = length(unique(Y))

	# assert(length(b)==(K1*(J-1)+K2)      ,"b has the wrong number of elements")
	# assert(ndims(b)==2   && size(b,2) ==1,"b must be a column vector")
	# assert(ndims(Y)==2   && size(Y,2) ==1,"Y must be a column vector")
	# assert(minimum(Y)==1 && maximum(Y)==J,"Y should contain integers numbered consecutively from 1 through J")
	# if ~isempty(X)
		# assert(ndims(X) ==2 ,"X must be a 2-dimensional matrix")
		# assert(size(X,1)==N ,"The 1st dimension of X should equal the number of observations in Y")
	# end
	# if ~isempty(Z)
		# assert(ndims(Z)==3  ,"Z must be a 3-dimensional array")
		# assert(size(Z,1)==N ,"The 1st dimension of Z should equal the number of observations in Y")
		# assert(size(Z,3)==J ,"The 3rd dimension of Z should equal the number of alternatives in Y")
	# end
	# if nargin>=7 && ~isempty(W)
		# assert(ndims(W)==2 && size(W,2)==1 && length(W)==N,"W must be a column vector the same size as Y")
	# end

	# apply restrictions as defined in restrMat
	if ~isempty(restrMat)
		b = applyRestr(restrMat,b)
	end

	# if nargin==5 || isempty(baseAlt)
		# baseAlt = J
	# end

	# if nargin==5 || nargin==6 || isempty(W)
		# W = ones(N,1)
	# end

	b2    = b[K1*(J-1)+1:K1*(J-1)+K2]
	num   = zeros(N,1)
	dem   = zeros(N,1)
	numer = ones(N,J)

	if K2>0 && K1>0 # hybrid of multinomial and conditional logit
		# sets BASEALT to be the alternative that is normalized to zero
		k = 1
		for j=setdiff(1:J,baseAlt)
			temp = X*b[(k-1)*K1+1:k*K1] + (Z[:,:,j]-Z[:,:,baseAlt])*b2
			num  = (Y.==j).*temp+num
			dem  = exp(temp)+dem
			numer[:,j] = exp(temp)
			k+=1
		end
		dem=dem+1
		P=numer./(dem*ones(1,J))
		
		like=-W'*(num-log(dem))
		
		# analytical gradient
		numg = zeros(K2,1)
		demg = zeros(K2,1)
		
		k = 1
		grad = zeros(size(b))
		for j=setdiff(1:J,baseAlt)
			grad[(k-1)*K1+1:k*K1]=-X'*(W.*((Y.==j)-P[:,j]))
			k+=1
		end
		k = 1
		for j=setdiff(1:J,baseAlt)
			numg = -(Z[:,:,j]-Z[:,:,baseAlt])'*(W.*(Y.==j))+numg
			demg = -(Z[:,:,j]-Z[:,:,baseAlt])'*(W.*P[:,j])+demg
			k+=1
		end
		grad[K1*(J-1)+1:K1*(J-1)+K2]=numg-demg
		
		# constraints for analytical gradient
		if ~isempty(restrMat)
			applyRestrGrad(restrMat,grad)
		end
		return like,grad
	elseif K1>0 && K2==0 # traditional multinomial logit
		# sets BASEALT to be the alternative that is normalized to zero
		k = 1
		for j=setdiff(1:J,baseAlt)
			temp = X*b[(k-1)*K1+1:k*K1]
			num  = (Y.==j).*temp+num
			dem  = exp(temp)+dem
			numer[:,j] = exp(temp)
			k+=1
		end
		dem=dem+1
		P=numer./(dem*ones(1,J))
		
		like=-W'*(num-log(dem))
		
		# analytical gradient
		k = 1
		grad = zeros(size(b))
		for j=setdiff(1:J,baseAlt)
			grad[(k-1)*K1+1:k*K1]=-X'*(W.*((Y.==j)-P[:,j]))
			k+=1
		end
		
		# constraints for analytical gradient
		if ~isempty(restrMat)
			applyRestrGrad(restrMat,grad)
		end
		return like,grad
	elseif K1==0 && K2>0 # traditional conditional logit
		# sets BASEALT to be the alternative that is normalized to zero
		k = 1
		for j=setdiff(1:J,baseAlt)
			temp = (Z[:,:,j]-Z[:,:,baseAlt])*b2
			num  = (Y.==j).*temp+num
			dem  = exp(temp)+dem
			numer[:,j] = exp(temp)
			k+=1
		end
		dem=dem+1
		P=numer./(dem*ones(1,J))
		
		like=-W'*(num-log(dem))
		
		# analytical gradient
		numg = zeros(K2,1)
		demg = zeros(K2,1)
		
		grad = zeros(size(b))
		k = 1
		for j=setdiff(1:J,baseAlt)
			numg = -(Z[:,:,j]-Z[:,:,baseAlt])'*(W.*(Y.==j))+numg
			demg = -(Z[:,:,j]-Z[:,:,baseAlt])'*(W.*P[:,j])+demg
			k+=1
		end
		grad[K1*(J-1)+1:K1*(J-1)+K2]=numg-demg
		
		# constraints for analytical gradient
		if ~isempty(restrMat)
			applyRestrGrad(restrMat,grad)
		end
		return like,grad
	end

end

function datagen()
	# clear all clc
	# delete mlogitSimHess.diary
	# diary  mlogitSimHess.diary
	tic()
	srand(1234)
	# rng(seed,'twister')

	N       = convert(Int64,1e5) #inputs to functions such as -ones- need to be integers!!!
	T       = 5
	J       = 5

	# generate the data
	X = [ones(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1)]
	Z = zeros(N*T,3,J)
	for j=1:J
		Z[:,:,j] = [3+randn(N*T,1) randn(N*T,1)-1 rand(N*T,1)];
	end

	# X coefficients
	b      = zeros(size(X,2),J)
	b[:,1] = [-0.15 0.10  0.50 0.10]
	b[:,2] = [-1.50 0.15  0.70 0.20]
	b[:,3] = [-0.75 0.25 -0.40 0.30]
	b[:,4] = [ 0.65 0.05 -0.30 0.40]
	b[:,5] = [ 0.75 0.10 -0.50 0.50]

	# Z coefficients
	bz = [.2;.5;.8];

	# generate choice probabilities
	u   = zeros(N*T,J)
	p   = zeros(N*T,J)
	dem = zeros(N*T,1)
	for j=1:J
		u[:,j] = X*b[:,j]+Z[:,:,j]*bz
		# u[:,j] = X*b[:,j]
		dem=exp(u[:,j])+dem
	end
	for j=1:J
		p[:,j] = exp(u[:,j])./dem
	end

	# use the choice probabilities to create the observed choices
	draw=rand(N*T,1)
	Y=(draw.<sum(p[:,1:end],2))+
	  (draw.<sum(p[:,2:end],2))+
	  (draw.<sum(p[:,3:end],2))+
	  (draw.<sum(p[:,4:end],2))+
	  (draw.<sum(p[:,5:end],2))
	# tabulate(Y)
	println("Time spent generating data: ", toc())

	bAns = b[:]-repmat(b[:,J],J,1)
	bAns = cat(1,bAns[1:(J-1)*size(X,2)],bz)
	size(bAns)

	wgts = rand(N*T,1)
	
	startval = .05*rand((J-1)*size(X,2)+size(Z,2),1)
	tic()
	l,g = clogit(startval,[],Y,X,Z,J,wgts)
	println("Time spent evaluating clogit: ", toc())
	
	# res = Optim.optimize(clogit_ll,startval,method=:l_bfgs,grtol=1e-6,show_trace=true)
	# println(res.minimum)
	# println(res.f_minimum)
	
	# opt = Opt(:LD_LBFGS,length(bAns))
	# xtol_rel!(opt,1e-6)
	# min_objective!(opt,clogitHess)
	# minf,minx,ret = optimize(opt,startval)
	
	# # Now estimate using mlogitBaseAltRestrict:
	# derivative_checker = false
	# if derivative_checker==true
		# o4Nu=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',0,'MaxIter',0,'TolX',1e-8,'Tolfun',1e-8,'GradObj','off','DerivativeCheck','off','FinDiffType','central')
		# o4An=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',0,'MaxIter',0,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on' ,'DerivativeCheck','off','FinDiffType','central')
		# [bstruc0,lstruc,e,o,gNum]=fminunc('clogit',startval,o4Nu,[],Y,X,Z)
		# [bstruc0,lstruc,e,o,gAna]=fminunc('clogit',startval,o4An,[],Y,X,Z)
		# dlmwrite ('gradient_checker.csv',[gNum zeros(size(gNum,1),1) gAna])
		# return
	# end

	# hessian_checker = false
	# if hessian_checker==true
		# o4Nu=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',0,'MaxIter',0,'TolX',1e-8,'Tolfun',1e-8,'GradObj','off','Hessian','off','DerivativeCheck','off','FinDiffType','central')
		# o4An=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',0,'MaxIter',0,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on' ,'Hessian','on' ,'DerivativeCheck','off','FinDiffType','central')
		# [bstruc0,~,~,~,gNum,hNum]=fminunc('clogitHess',startval,o4Nu,[],Y,X,Z,[],wgts)
		# [bstruc0,~,~,~,gAna,hAna]=fminunc('clogitHess',startval,o4An,[],Y,X,Z,[],wgts)
		# hNum = full(hNum) hAna = full(hAna)
		# dlmwrite ('hessian_checker.csv',[hNum zeros(size(hNum,1),1) hAna])
		# return
	# end

	# # options=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',2000000,'MaxIter',15000,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on','DerivativeCheck','off','FinDiffType','central')
	# options=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',2000000,'MaxIter',15000,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on','Hessian','on','DerivativeCheck','off','FinDiffType','central')
	# startval = .05*rand((J-1)*size(X,2)+size(Z,2),1)
	# tic
	# bEst = fminunc('clogitHess',startval,options,[],Y,X,Z,[],wgts)
	# [bEst bAns]
	# disp(['Time spent on gradient + Hessian optimization: ',num2str(toc),' seconds'])
	# P = pclogit(bEst,Y,X,Z)
	# summarize(P)

	# options=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',2000000,'MaxIter',15000,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on','Hessian','off','DerivativeCheck','off','FinDiffType','central')
	# startval = .05*rand((J-1)*size(X,2)+size(Z,2),1)
	# tic
	# bEst = fminunc('clogitHess',startval,options,[],Y,X,Z,[],wgts)
	# [bEst bAns]
	# disp(['Time spent on gradient + Hessian optimization: ',num2str(toc),' seconds'])
	# P = pclogit(bEst,Y,X,Z)
	# summarize(P)
	# diary off
	
	# return X,Y,Z,wgts,bAns
	# return l,g,h,X,Y,Z,wgts,bAns
	
	# save("simData.jld",X,Y,Z,wgts,bAns)
end

@time datagen()
# close(f)
