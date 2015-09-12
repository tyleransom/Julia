## Simple simulation of mlogit
#
# Tyler Ransom
# Duke, June 25, 2015

# using Logging
using NLopt
using Optim
using JuMP
# using Debug
# using HDF5
# using JLD


function datagen()
	# clear all clc
	# delete mlogitSimHess.diary
	# diary  mlogitSimHess.diary
	# tic()
	srand(1234)

	N = convert(Int64,1e3) #inputs to functions such as -ones- need to be integers!!!
	T = 5
	# J = 5

	# generate the data
	const X = [ones(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) 15+3*randn(N*T,1) .7-.1*randn(N*T,1)]

	# X coefficients
	const bAns = [ 2.15, 0.10,  0.50, 0.10, 0.75, 1.2 ]
	
	# other inputs
	const restrMat = []
	const W = ones(size(X,1))
	const d = 1

	# create the observed outcomes
	draw = .3*randn(N*T,1)
	const Y = X*bAns+draw
	
	return X,Y,bAns,W,d,restrMat
end

include("normalMLEsimple.jl")

# function normalMLE(b::Vector{Float64})
	# @assert isa(b,Vector)
	# J = length(unique(d))

	# # apply restrictions as defined in restrMat
	# # if ~isempty(restrMat)
		# # b = applyRestr(restrMat,b)
	# # end

	# # slice parameter vector
	# beta      = b[1:end-J]
	# wagesigma = b[end-(J-1):end]

	# # log likelihood
	# likemat = zeros(length(Y),J)
	# dmat    = zeros(length(Y),J)
	# for j=1:J
		# dmat[:,j] = d.==j
		# likemat[:,j] = -.5*(log(2*pi)+log(wagesigma[j]^2)+((Y-X*beta)./wagesigma[j]).^2)
	# end
	# like = -sum(W.*sum(dmat.*likemat,2))
	# return like
# end

function estimation()
	#------------
	# Estimation
	#------------
	# OLS
	bhat = X\Y
	
	startval = vec(rand(length(bAns)+1))
	# # MLE with Optim package
	# @assert isa(startval,Vector)
	# normalMLEsimple(startval)
	# # # conjugate gradient
	# # optimum = optimize(normalMLEsimple,normalMLEsimplegradient!,startval,method=:cg,show_trace=false,grtol=1e-6,xtol=1e-6,ftol=1e-6,iterations=1000000)
	# # bhatMLE1 = optimum.minimum
	# # # bfgs
	# # optimum = optimize(normalMLEsimple,normalMLEsimplegradient!,startval,method=:bfgs,show_trace=false,grtol=1e-6,xtol=1e-6,ftol=1e-6,iterations=1000000)
	# # bhatMLE2 = optimum.minimum
	# # l-bfgs, autodiff
	# println("autodiff")
	# optimum = @time optimize(normalMLEsimple,startval,method=:l_bfgs,show_trace=false,autodiff=true,grtol=1e-6,xtol=1e-6,ftol=1e-6,iterations=1000000)
	# bhatMLE3a = optimum.minimum
	# # l-bfgs, no autodiff and no gradient
	# println("no autodiff, no gradient")
	# optimum = @time optimize(normalMLEsimple,startval,method=:l_bfgs,show_trace=false,autodiff=true,grtol=1e-6,xtol=1e-6,ftol=1e-6,iterations=1000000)
	# bhatMLE3b = optimum.minimum
	# l-bfgs (d4 implementation)
	println("analytical gradient")
	optimum = @time optimize(d4,startval,method=:l_bfgs,show_trace=false,grtol=1e-6,xtol=1e-6,ftol=1e-6,iterations=1000000)
	bhatMLE4 = optimum.minimum
	# # # simulated annealing
	# # optimum = optimize(normalMLEsimple,normalMLEsimplegradient!,startval,method=:simulated_annealing,show_trace=false,grtol=1e-6,xtol=1e-6,ftol=1e-6,iterations=1000000)
	# # bhatMLE4 = optimum.minimum
	# # # nelder mead
	# # optimum = optimize(normalMLEsimple,normalMLEsimplegradient!,startval,method=:nelder_mead,show_trace=false,grtol=1e-6,xtol=1e-6,ftol=1e-6,iterations=1000000)
	# # bhatMLE5 = optimum.minimum
	# println(bhat)
	# # println(bhatMLE1')
	# # println(bhatMLE2')
	# println(bhatMLE3a')
	# println(bhatMLE3b')
	println(bhatMLE4')
	# # println(bhatMLE5')
	
	# MLE with NLopt package
	opt = Opt(:LD_LBFGS,length(startval))
	xtol_rel!(opt,1e-6)
	ftol_rel!(opt,1e-6)
	ftol_rel!(opt,1e-6)
	min_objective!(opt,normalMLEsimpleNLopt)
	minf,minx,ret = optimize!(opt,startval)
	println(minx)
end

function estimation2()
	#------------
	# Estimation
	#------------
	# OLS
	bhat = X\Y
	
	# MLE with NLopt package
	startval = vec(rand(length(bAns)+1))
	# nelder-mead
	opt = Opt(:LN_NELDERMEAD,length(startval))
	xtol_rel!(opt,1e-6)
	ftol_rel!(opt,1e-6)
	ftol_rel!(opt,1e-6)
	min_objective!(opt,normalMLEsimpleNLopt)
	minf,minx,ret = optimize!(opt,startval)
	println(minx)
end

@time X,Y,bAns,W,d,restrMat = datagen()
# @time estimation2()
@time estimation()
# close(f)
