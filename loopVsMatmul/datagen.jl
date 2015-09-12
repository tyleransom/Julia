function datagen(N::Int)
	X    = [ones(N,1) randn(N,1) 8*rand(N,1)-4 100*randn(N,1)+40 rand(N,1)];
	beta = randn(size(X,2),1);
	return X,beta
end