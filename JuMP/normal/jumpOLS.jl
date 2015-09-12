function jumpOLS()
	## OLS of classical linear regression model
	# myOLS = Model(solver=NLoptSolver(algorithm=:LN_NELDERMEAD))
	myOLS = Model(solver=IpoptSolver(tol=1e-8))
	
	@defVar(myOLS, b[i=1:16], start = bAns[i])
	
	@setNLObjective(myOLS, Min, sum{(Y[i]-sum{X[i,k]*b[k], k=1:16})^2, i=1:n})
	
	solve(myOLS)
	
	println("beta = ", getValue(b[:]))
	SSE = sum((Y-X*getValue(b[:])).^2)
	s = sqrt(SSE/(n-size(X,2)))
	println("s = ", s)
	println("OLS objective: ", getObjectiveValue(myOLS))
	
	#other JuMP returns:
	println("OLS sense: ", getObjectiveSense(myOLS))
end