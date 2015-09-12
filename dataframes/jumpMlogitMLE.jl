function jumpMlogitMLE(Y,X,baseAlt=maximum(Y))
	## MLE of multinomial logit model
	# Declare the name of your model and the optimizer you will use
	myMLE = Model(solver=IpoptSolver(tol=1e-8,print_level=0))
	
	n  = size(X,1)
	K1 = size(X,2)
	J  = maximum(Y)
	# baseAlt = J
	
	# re-express baseAlt as last alternative
	if baseAlt!=J
		Y[Y.==J] = J+1
		Y[Y.==baseAlt] = J
		Y[Y.==J+1] = baseAlt
	end
	
	# Declare the variables you are optimizing over
	@defVar(myMLE, bx[k=1:K1,j=1:J-1])
	# @defVar(myMLE, bz[k=1:K2])
	
	# Write constraints here if desired (in this case, set betas for base alternative to be 0)
	# for k=1:K1
	  # @addConstraint(myMLE, bx[k,baseAlt] == 0)
	# end
	
	# Write the objective function
	@defNLExpr(likelihood, sum{ sum{ (Y[i]==j)*(sum{ X[i,k]*(bx[k,j]),k=1:K1} ) ,j=1:J-1} - log(1 + sum { exp( sum{ X[i,k]*(bx[k,j]),k=1:K1} ) ,j=1:J-1} ) ,i=1:n});
	
	# Set the objective function to be maximized
	@setNLObjective(myMLE, Max, likelihood )
	
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
	bxOpt = getValue(bx[:,:])
	# bzOpt = getValue(bz[:])
	seOptx = reshape(seOpt[1:length(vec(bxOpt))],size(bxOpt))'
	
	# correct re-expression of baseAlt as last alternative
	if baseAlt!=J
		bxOpt = bxOpt[:,[setdiff(1:J-1,baseAlt),baseAlt]]
		
		# if baseAlt==1
			# bxOpt = bxOpt[:,[2,3,4,baseAlt]]
		# elseif baseAlt==2
			# bxOpt = bxOpt[:,[1,3,4,baseAlt]]
		# Y[Y.==J] = J+1
		# Y[Y.==baseAlt] = J
		# Y[Y.==J+1] = baseAlt
	end
	
	# Print estimates and log likelihood value
	println("beta = ", bxOpt[:,:])
	println("MLE objective: ", getObjectiveValue(myMLE))
	println("MLE status: ", status)
	return bxOpt,seOptx
end