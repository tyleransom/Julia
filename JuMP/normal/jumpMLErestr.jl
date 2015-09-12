function jumpMLErestr()
	## MLE of classical linear regression model
	# Declare the name of your model and the optimizer you will use
	# myMLE = Model(solver=NLoptSolver(algorithm=:LN_NELDERMEAD)) # can also change to other NLopt optimizers (e.g. :LN_BOBYQA, :LD_LBFGS, etc.)
	myMLE = Model(solver=IpoptSolver(tol=1e-8))
	
	# Declare the variables you are optimizing over
	@defVar(myMLE, b[i=1:16], start = bAns[i])
	@defVar(myMLE, s >=0.0, start = sigAns)
	# @defVar(myMLE, s ==0.5) # use this syntax if you want to restrict a parameter to a specified value. Note that NLopt has issues with equality constraints
	
	# Write constraints here if desired
	@addConstraint(myMLE, b[15] == 0)
	
	# Write your objective function
	@setNLObjective(myMLE, Max, (n/2)*log(1/(2*pi*s^2))-sum{(Y[i]-sum{X[i,k]*b[k], k=1:16})^2, i=1:n}/(2s^2))
	
	# Solve the objective function
	status = solve(myMLE)
	
	# Generate Hessian
	this_par     = myMLE.colVal
	m_const_mat  = JuMP.prepConstrMatrix(myMLE)
	m_eval       = JuMP.JuMPNLPEvaluator(myMLE, m_const_mat);
	MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])
	grad = fill(0., length(this_par))
	MathProgBase.eval_grad_f(m_eval, grad, this_par)
	hess_struct  = MathProgBase.hesslag_structure(m_eval)
	hess_vec     = zeros(length(hess_struct[1]))
	numconstr    = length(m_eval.m.linconstr) + length(m_eval.m.quadconstr) + length(m_eval.m.nlpdata.nlconstr)
	dimension    = length(myMLE.colVal)
	MathProgBase.eval_hesslag(m_eval, hess_vec, this_par, 1.0, zeros(numconstr))
	this_hess_ld = sparse(hess_struct[1], hess_struct[2], hess_vec, dimension, dimension)
	hOpt         = this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)));
	hOpt         = -full(hOpt) #since we are maximizing
	
	# Calculate standard errors
	varOpt = hOpt\(-grad*-grad')/hOpt  #take negative of gradient since we are maximizing
	# seOpt = sqrt(diag(full(-1*hOpt)\eye(size(hOpt,1))))
	# seOpt = rand(size(hOpt,1))
	seOpt = sqrt(diag(varOpt))
	
	# Save estimates
	bOpt = getValue(b[:])
	sOpt = getValue(s)
	
	# Print estimates and log likelihood value
	# println("beta = ", bOptInterim)
	# println("s = ", sOptInterim)
	println("beta = ", bOpt)
	println("s = ", sOpt)
	println("MLE objective: ", getObjectiveValue(myMLE))
	println("MLE status: ", status)
	return bOpt,sOpt,grad,hOpt,varOpt,seOpt
end