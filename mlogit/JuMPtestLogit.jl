#  Copyright 2015, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
using JuMP
using NLopt

function datagen()
	srand(1234)

	N = convert(Int64,1e5) #inputs to functions such as -ones- need to be integers!!!
	T = 5
	const n = convert(Int64,N*T)
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
	
	# return draw
	return X,Y,bAns,W,d,restrMat,n
end

function jumperVec()
	myMLE = Model(solver=NLoptSolver(algorithm=:LD_LBFGS))
	# myMLE = Model(solver=NLoptSolver(algorithm=:LN_BOBYQA))

	@defVar(myMLE, b1, start = 0.2)
	@defVar(myMLE, b2, start = 0.1)
	@defVar(myMLE, b3, start = 0.5)
	@defVar(myMLE, b4, start = 0.8)
	@defVar(myMLE, b5, start = 0.3)
	@defVar(myMLE, b6, start = 0.4)
	@defVar(myMLE, s >=0.0, start = 1.0)

	@setNLObjective(myMLE, Max, (n/2)*log(1/(2*pi*s^2))-sum{(Y[i]-X[i,1]*b1-X[i,2]*b2-X[i,3]*b3-X[i,4]*b4-X[i,5]*b5-X[i,6]*b6)^2, i=1:n}/(2s^2))
	
	# @addNLConstraint(myMLE, b6 == s)

	solve(myMLE)

	println("beta[1] = ", getValue(b1))
	println("beta[2] = ", getValue(b2))
	println("beta[3] = ", getValue(b3))
	println("beta[4] = ", getValue(b4))
	println("beta[5] = ", getValue(b5))
	println("beta[6] = ", getValue(b6))
	# println("mean(data) = ", mean(data))
	println("s = "      , getValue(s))
	# println("var(data) = ", var(data))
	println("MLE objective: ", getObjectiveValue(myMLE))

	# # constrained MLE?

	# solve(myMLE)
	# println("\nWith constraint µ == s^2:")
	# println("µ = ", getValue(µ))
	# println("s^2 = ", getValue(s)^2)

	# println("Constrained MLE objective: ", getObjectiveValue(myMLE))
	# # return Y,X,bAns
end

@time X,Y,bAns,W,d,restrMat,n = datagen()
# @time jumper()
Y,X,bAns = @time jumperVec()