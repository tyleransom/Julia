function jumpMLE(Y,X)
	## MLE of classical linear regression model
	# Declare the name of your model and the optimizer you will use
	# myMLE = Model(solver=NLoptSolver(algorithm=:LN_NELDERMEAD)) # can also change to other NLopt optimizers (e.g. :LN_BOBYQA, :LD_LBFGS, etc.)
	myMLE = Model(solver=IpoptSolver(tol=1e-6,print_level=0))
	
	K = size(X,2)
	n = size(X,1)
	# Declare the variables you are optimizing over
	@defVar(myMLE, b[i=1:K])
	@defVar(myMLE, s >=0.0)
	# @defVar(myMLE, s ==0.5) # use this syntax if you want to restrict a parameter to a specified value. Note that NLopt has issues with equality constraints
	
	# Write constraints here if desired
	# @addNLConstraint(m, µ == s^2)
	
	# Write your objective function
	@setNLObjective(myMLE, Max, (n/2)*log(1/(2*pi*s^2))-sum{(Y[i]-sum{X[i,k]*b[k], k=1:K})^2, i=1:n}/(2s^2))
	
	# Solve the objective function
	status = solve(myMLE)
	
	# Generate Hessian
	this_par = myMLE.colVal
	m_const_mat = JuMP.prepConstrMatrix(myMLE)
	m_eval = JuMP.JuMPNLPEvaluator(myMLE);
	MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])
	hess_struct = MathProgBase.hesslag_structure(m_eval)
	hess_vec = zeros(length(hess_struct[1]))
	numconstr = length(m_eval.m.linconstr) + length(m_eval.m.quadconstr) + length(m_eval.m.nlpdata.nlconstr)
	dimension = length(myMLE.colVal)
	MathProgBase.eval_hesslag(m_eval, hess_vec, this_par, 1.0, zeros(numconstr))
	this_hess_ld = sparse(hess_struct[1], hess_struct[2], hess_vec, dimension, dimension)
	hOpt = this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)));
	hOpt = -full(hOpt) #since we are maximizing
	
	# Calculate standard errors
	seOpt = sqrt(diag(full(hOpt)\eye(size(hOpt,1))))
	
	# Save estimates
	bOpt = getValue(b[:])
	sOpt = getValue(s)
	
	# Print estimates and log likelihood value
	println("beta = ", bOpt)
	println("s = ", sOpt)
	println("MLE objective: ", getObjectiveValue(myMLE))
	println("MLE status: ", status)
	return bOpt,sOpt,hOpt,seOpt
end