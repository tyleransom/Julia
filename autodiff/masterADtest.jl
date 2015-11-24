# This script tests auto differentiation using a simple OLS objective function

# using DualNumbers
using ForwardDiff
# using HyperDualNumbers
# using ReverseDiffSource
# using TaylorSeries
# using Calculus
# using PowerSeries
# using ReverseDiffSparse

include("datagen.jl")
include("olsobjfun.jl")
include("olsobjfunIndiv.jl")

# Generate data and evaluate objective functions:
@time X,Y,bAns,sigAns,n = datagen()
err    = olsobjfun(bAns)
errmat = olsobjfunIndiv(bAns)

# Calculate gradient
@time grad = ForwardDiff.gradient(olsobjfun,bAns)
@time gradIndiv = ForwardDiff.jacobian(olsobjfunIndiv,bAns)
# # compare analytical gradient with ForwardDiff.gradient:
# # println(cat(2,-2*X'*(Y-X*bAns),grad))
# # Calculate hessian
# @time hess = ForwardDiff.hessian(olsobjfun,bAns)
# # compare analytical gradient with ForwardDiff.gradient:
# # println(cat(1,2*X'*X,hess))
# # hess = ForwardDiff.hessian(olsobjfun,bAns)
# # compare analytical gradient with ForwardDiff.gradient:
indivGrad = zeros(n,length(bAns))
for i=1:n
	indivGrad[i,:] = -2*X[i,:]'*(Y[i]-X[i,:]*bAns)
end
# # jac  = ForwardDiff.jacobian(olsobjfun,grad)
# # println(grad)
# # size(jac)