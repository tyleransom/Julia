## Simple simulation of mlogit
#
# Tyler Ransom
# Duke, June 25, 2015

# using Logging
# using NLopt
using Optim
# using Debug
# using HDF5
# using JLD


function datagen()
	# clear all clc
	# delete mlogitSimHess.diary
	# diary  mlogitSimHess.diary
	# tic()
	srand(1234)

	N = convert(Int64,1e5) #inputs to functions such as -ones- need to be integers!!!
	T = 5
	# J = 5

	# generate the data
	const X = [ones(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) 15+3*randn(N*T,1) .7-.1*randn(N*T,1)]

	# X coefficients
	const bAns = [ 2.15 0.10  0.50 0.10 0.75 1.2 ]'
	
	# other inputs
	const restrMat = []
	const W = ones(size(X,1),1)
	const d = 1

	# create the observed outcomes
	draw = .3*randn(N*T,1)
	const Y = X*bAns+draw
	
	return X,Y,bAns,W,d,restrMat
end

function normalMLE(b)
	# apply restrictions as defined in restrMat
	# if ~isempty(restrMat)
		# b = applyRestr(restrMat,b)
	# end

	# slice parameter vector
	beta      = b[1:end-1]
	wagesigma = b[end]

	# log likelihood
	like = -W'*(-.5*(log(2*pi)+log(wagesigma^2)+((Y-X*beta)./wagesigma).^2))
	return like
end

# function normalMLEgradient!(b::Vector{Float64},out::Vector{Float64})
	# J = length(unique(d))

	# # slice parameter vector
	# beta      = b[1:end-J]
	# wagesigma = b[end-(J-1):end]
	# n         = length(Y

	# # analytical gradient
	# for j=1:J
		# out[1:end-J] += -X'*(W.*(d.==j).*(Y-X*beta)./(wagesigma[j].^2))
	# end
	# for j=1:J
		# k=length(b)-(J-1)+j-1
		# temp = 1./wagesigma[j]-((Y-X*beta).^2)./(wagesigma[j].^3)
		# out[k] = sum(W.*(d.==j).*temp)
	# end

# end

function estimation()
	#------------
	# Estimation
	#------------
	# OLS
	bhat = X\Y
	println(bhat)
	
	# MLE with Optim package
	startval = rand(length(bAns)+1,1)
	normalMLE(startval)
	# # optimum = optimize(normalMLE,normalMLEgradient!,startval,method=:cg)
	optimum = optimize(normalMLE,startval,method=:cg)
	bhatMLE = optimum.minimum
	println(bhatMLE)
	
	# MLE with NLopt package
	# opt = Opt(:LD_LBFGS,length(bAns))
	# xtol_rel!(opt,1e-6)
	# min_objective!(opt,normalMLE)
	# minf,minx,ret = optimize(opt,startval)
end

@time X,Y,bAns,W,d,restrMat = datagen()
@time estimation()
# close(f)
