function JuMPxtlogitMLE()
	## MLE of multinomial logit model
	# Declare the name of your model and the optimizer you will use
	myMLE = Model(solver=IpoptSolver(tol=1e-6))
	
	# Declare the variables you are optimizing over
	@defVar(myMLE, bx[k=1:K1])
	# @defVar(myMLE, bx[k=1:K1,j=setdiff(1:J,2)])
	@defVar(myMLE, sig)
	
	# # Write constraints here if desired (in this case, set betas for base alternative to be 0)
	# for k=1:K1
	  # @addConstraint(myMLE, bx[k,2] == 0)
	# end
	
	#create grid (from N(0,sig)) draws
	const G = 15
	grid,quad_weight = lgwt(G,-11,11);
	const grid = grid
	const gridDen = pdf(Normal(0,1),grid)
	const quad_weight = quad_weight
	
	# Write the objective function
	@defNLExpr(likelihood, sum{ log( sum{ prod { (exp( sum{ X[i,t,k]*(bx[k]),k=1:K1} + sig*grid[g] )/(1+exp( sum{ X[i,t,k]*(bx[k]),k=1:K1} + sig*grid[g] )))^((Y[i,t]==1)) *
	(1/(1+exp( sum{ X[i,t,k]*(bx[k]),k=1:K1} + sig*grid[g] )))^((Y[i,t]==2)),t=1:T}  * gridDen[g] * quad_weight[g] ,g=1:G}) ,i=1:N});
	
	# Set the objective function to be maximized
	@setNLObjective(myMLE, Max, likelihood )
	
	# Solve the objective function
	status = solve(myMLE)
	
	# Generate Hessian
	this_par = myMLE.colVal
	m_const_mat = JuMP.prepConstrMatrix(myMLE)
	m_eval = JuMP.JuMPNLPEvaluator(myMLE, m_const_mat);
	MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])
	hess_struct = MathProgBase.hesslag_structure(m_eval)
	hess_vec = zeros(length(hess_struct[1]))
	numconstr = length(m_eval.m.linconstr) + length(m_eval.m.quadconstr) + length(m_eval.m.nlpdata.nlconstr)
	dimension = length(myMLE.colVal)
	MathProgBase.eval_hesslag(m_eval, hess_vec, this_par, 1.0, zeros(numconstr))
	this_hess_ld = sparse(hess_struct[1], hess_struct[2], hess_vec, dimension, dimension)
	# println(full(this_hess_ld))
	hOpt = this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)));
	hOpt = -full(hOpt) #since we are maximizing
	# println(full(this_hess))
	
	# Calculate standard errors
	seOpt = sqrt(diag(hOpt\eye(size(hOpt,1))))
	# # swap indices because order of coefficients is different than ordering of Hessian rows
	# # note: it would be better to do this re-ordering to the Hessian, but I can't figure out...
	# # how to do this simultaneously for rows and columns
	# ind = [1:size(hOpt,1)]
	# for j=1:J-1
		# ind[(j-1)*K1+1:j*K1] = [j:K1:(J-1)*K1]
	# end
	# seOpt = seOpt[ind]
	
	# Save estimates
	bxOpt  = getValue(bx[:])
	sigOpt = getValue(sig)
	# hOpt  = -eye(length(bx)+length(bz))
	
	# Print estimates and log likelihood value
	println("beta = ", bxOpt[:])
	println("sig = ", sigOpt)
	println("MLE objective: ", getObjectiveValue(myMLE))
	println("MLE status: ", status)
	return bxOpt,sigOpt,hOpt,seOpt
end