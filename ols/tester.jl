using Distributions
using Debug

srand(2)
N=convert(Int64,1e5)
K=3
genX = MvNormal(eye(K))
X = rand(genX,N)
X = X'
X_noconstant = X
constant = ones(N)
X = [constant X]
W = ones(N)

genEpsilon = Normal(0, 1)
epsilon = rand(genEpsilon,N)
trueParams = [0.1,0.5,-0.3,0.]
Y = X*trueParams + epsilon

function OLSestimator(y,x)
    estimate = inv(x'*x)*(x'*y)
    return estimate
end

function loglike(rho)
   beta = rho[1:4]
   sigma2 = exp(rho[5])
   residual = Y-X*beta
   dist = Normal(0, sqrt(sigma2))
   contributions = logpdf(dist,residual)
   loglikelihood = sum(contributions)
   return -loglikelihood
end

function normalMLE(b)
	# slice parameter vector
	beta      = b[1:end-1]
	wagesigma = b[end]

	# log likelihood
	like = -sum(vec(-.5*(log(2*pi)+log(wagesigma^2)+((Y-X*beta)./wagesigma).^2)))
	return like
end

estimates = @time OLSestimator(Y,X)
println(estimates)
est2 = @time linreg(X_noconstant,Y)
println(est2)

using Optim
params0 = [.1,.2,.3,.4,.5]
# # conjugate-gradient
# optimum = optimize(loglike,params0,method=:cg,show_trace=true)
# @time MLE = optimum.minimum
# MLE[5] = exp(MLE[5])
# println(MLE)
# # l-bfgs
# optimum = optimize(loglike,params0,method=:l_bfgs,show_trace=true)
# @time MLE = optimum.minimum
# MLE[5] = exp(MLE[5])
# println(MLE)
# # nelder-mead
# optimum = optimize(loglike,params0,method=:nelder_mead,show_trace=true)
# @time MLE = optimum.minimum
# MLE[5] = exp(MLE[5])
# println(MLE)
# # simulated annealing
# optimum = optimize(loglike,params0,method=:nelder_mead,show_trace=true)
# @time MLE = optimum.minimum
# MLE[5] = exp(MLE[5])
# println(MLE)
# evaluate my function
normalMLE(params0)
# try to optimize my function
optimum = optimize(normalMLE,params0,method=:cg,show_trace=true,grtol=1e-6)
@time MLE = optimum.minimum
println(MLE)
optimum = optimize(normalMLE,params0,method=:l_bfgs,show_trace=true,grtol=1e-6)
@time MLE = optimum.minimum
println(MLE)
