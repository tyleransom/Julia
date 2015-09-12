function normalMLE(b::Vector{Float64})
	# apply restrictions as defined in restrMat
	if ~isempty(restrMat)
		b = applyRestr(restrMat,b)
	end

	# slice parameter vector
	beta      = b[1:end-J]
	wagesigma = b[end-(J-1):end]

	# log likelihood
	likemat = zeros(length(Y),J)
	dmat    = zeros(length(Y),J)
	for j=1:J
		dmat[:,j] = d==j
		likemat[:,j] = -.5*(log(2*pi)+log(wagesigma[j]^2)+((Y-X*beta)./wagesigma[j]).^2)
	end
	like = -W'*sum(dmat.*likemat,2)

end


function normalMLEsimplegradient!(b::Vector{Float64},out::Vector{Float64})
	# analytical gradient
	out = zeros(length(b))
	for j=1:J
		out[1:end-J] += -X'*(W.*(d.==j).*(Y-X*beta)./(wagesigma[j].^2))
	end
	for j=1:J
		k=length(b)-(J-1)+j-1
		temp = vec(1./wagesigma[j]-((Y-X*beta).^2)./(wagesigma[j].^3))
		out[k] = sum(W.*(d.==j).*temp)
	end

end