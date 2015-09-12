function bootstrapper(B)
	bootEst = zeros(size(x,2)+1,B);
	fvalEst = zeros(B,1)
	for b=1:B
		idx = sample(1:n,n)
		# if b==2
			# println(cat(2,sort(idx),[1:n]))
		# end
		global X = x[idx,:]
		global Y = y[idx,:]
		@assert ~isequal(idx,1:n)
		@assert ~isequal(X,x)
		@assert ~isequal(Y,y)
		bOpt,sOpt,fOpt = jumpMLErestrBoot()
		bootEst[:,b] = cat(1,bOpt,sOpt)''
		fvalEst[b] = fOpt
	end
	bOptBoot = mean(bootEst,2)
	seOptBoot = std(bootEst,2)
	return bOptBoot,seOptBoot,bootEst,fvalEst
end