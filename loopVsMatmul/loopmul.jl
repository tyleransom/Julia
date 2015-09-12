function loopmul(X,beta)
	Y = zeros(size(X,1),1)
	for i=1:size(X,1)
		for j=1:size(X,2)
			Y[i]+=X[i,j]*beta[j]
		end
	end
	return Y
end