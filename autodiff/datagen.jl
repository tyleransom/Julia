function datagen()
	## Generate data for a linear model to test optimization
	srand(1234)
	
	N = convert(Int64,1e5) #inputs to functions such as -ones- need to be integers!
	T = 5
	const n = convert(Int64,N*T)
	
	# generate the Xs
	const X = [ones(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) 15+3*randn(N*T,1) .7-.1*randn(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) 15+3*randn(N*T,1) .7-.1*randn(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) 15+3*randn(N*T,1) .7-.1*randn(N*T,1)]
	
	# generate the betas (X coefficients)
	const bAns = [ 2.15, 0.10,  0.50, 0.10, 0.75, 1.2, 0.10,  0.50, 0.10, 0.75, 1.2, 0.10,  0.50, 0.10, 0.75, 1.2 ]
	
	# generate the std dev of the errors
	sigAns = 0.3
	
	# generate the Ys from the Xs, betas, and error draws
	draw = 0 + sigAns*randn(N*T,1)
	const Y = X*bAns+draw
	
	# return generated data so that other functions (below) have access
	return X,Y,bAns,sigAns,n
end