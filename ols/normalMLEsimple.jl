function normalMLEsimple{T}(b::Vector{T})
	# slice parameter vector
	beta      = b[1:end-1]
	wagesigma = b[end]

	# log likelihood
	like = -sum(vec(-.5*(log(2*pi)+log(wagesigma^2)+((Y-X*beta)./wagesigma).^2)))
	return like
end

function normalMLEsimplegradient!(b::Vector{Float64},out::Vector{Float64})
	# slice parameter vector
	beta      = b[1:end-1]
	wagesigma = b[end]

	# analytical gradient
	out[1:end-1] = -X'*((Y-X*beta)./(wagesigma.^2))
	out[end] = sum(1./wagesigma-((Y-X*beta).^2)./(wagesigma.^3))
end

function normalMLEsimpleAndGradient!(b::Vector{Float64},out::Vector{Float64})
	# slice parameter vector
	beta      = b[1:end-1]
	wagesigma = b[end]

	# log likelihood
	like = -sum(vec(-.5*(log(2*pi)+log(wagesigma^2)+((Y-X*beta)./wagesigma).^2)))
	
	# analytical gradient
	out[1:end-1] = -X'*((Y-X*beta)./(wagesigma.^2))
	out[end] = sum(1./wagesigma-((Y-X*beta).^2)./(wagesigma.^3))
	return like
end

d4 = DifferentiableFunction(normalMLEsimple,normalMLEsimplegradient!,normalMLEsimpleAndGradient!)

function normalMLEsimpleNLopt(b::Vector,grad::Vector)
	# slice parameter vector
	beta      = b[1:end-1]
	wagesigma = b[end]

	# log likelihood
	like = -sum(vec(-.5*(log(2*pi)+log(wagesigma^2)+((Y-X*beta)./wagesigma).^2)))
	
	# analytical gradient
	grad[1:end-1] = -X'*((Y-X*beta)./(wagesigma.^2))
	grad[end]     = sum(1./wagesigma-((Y-X*beta).^2)./(wagesigma.^3))
	
	return like
end