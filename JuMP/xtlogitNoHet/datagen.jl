function datagen()
	## Generate data for a linear model to test optimization
	srand(1234)
	
	const N = convert(Int64,1e5) #inputs to functions such as -ones- need to be integers!
	const T = 5
	const J = 2
	const n = convert(Int64,N*T) # use -const- as a way to declare a variable to be global (so other functions can access it)

	# generate the covariates
	X = cat(3,ones(N,T),5+3*randn(N,T),rand(N,T),2.5+2*randn(N,T))
	const K1 = size(X,3)
	Z = zeros(N,T,3,J)
	for j=1:J
		Z[:,:,:,j] = cat(3,3+randn(N,T),randn(N,T)-1,rand(N,T));
	end
	const K2 = size(Z,3)
	const baseAlt = 2

	# X coefficients
	bxAns      = zeros(K1,J)
	bxAns[:,1] = [-0.15 0.10  0.50 0.10 ]
	bxAns[:,2] = [-1.50 0.15  0.70 0.20 ]

	# Z coefficients
	bzAns = [.2;.5;.8]
	
	# random effect variance
	sigAns = .35
	
	# generate choice probabilities
	u      = zeros(N,T,J)
	p      = zeros(N,T,J)
	dem    = zeros(N,T)
	factor = zeros(N,1);
	for j=1:J
		for t=1:T
			u[:,t,j] = squeeze(X[:,t,:],2)*bxAns[:,j]+squeeze(squeeze(Z[:,t,:,j],4),2)*bzAns + factor
			# u[:,j] = X*b[:,j]
		end
		dem=exp(u[:,:,j])+dem
	end
	
	for j=1:J
		# for t=1:T
			p[:,:,j] = exp(u[:,:,j])./dem
		# end
	end
	
	# use the choice probabilities to create the observed choices
	draw=rand(N,T)
	Y=(draw.<sum(p[:,:,1:end],3))+
	  (draw.<sum(p[:,:,2:end],3))
	
	# return generated data so that other functions (below) have access
	return X,Y,Z,bxAns,bzAns,sigAns,n,N,T,J,K1,K2,baseAlt
end