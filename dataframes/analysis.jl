function analysis()
	data = readtable("auto.csv") #insheet
	data[:logprice] = log(data[:price]) # gen logprice = log(price)
	dump(data) # like stata's describe

	OLS1 = lm(price~mpg+weight+gear_ratio, data)
	OLS2 = lm(logprice~mpg+weight+gear_ratio,data)
	OLS3 = glm(logprice~mpg+weight+gear_ratio,data,Normal(),IdentityLink(),wts=data[:headroom].data)
	logit1= glm(foreign~mpg+weight+gear_ratio,data,Binomial(),LogitLink())
	logit2= glm(foreign~mpg+weight+gear_ratio,data,Binomial(),LogitLink(),wts=data[:headroom].data)
	println(OLS1)
	println(OLS2)
	println(OLS3)
	println(logit1)
	println(logit2)

	# X1 = convert(Array,[data[:mpg] data[:weight] data[:gear_ratio]])
	# X = cat(2,ones(size(X1,1)),X1)
	# Y = convert(Array,[data[:logprice]])
	# bhat,shat = @time jumpMLE(Y,X)
	# println([bhat,shat]')
	
	# Drop missings and convert to regular Julia matrix for optimization
	deleterows!(data,find(isna(data[:rep78]))) # drop if mi(rep78)
	X1 = convert(Array,[data[:mpg] data[:weight] data[:gear_ratio]])
	X = cat(2,ones(size(X1,1)),X1)
	Y = convert(Array,[data[:rep78]])
	# mlogisticreg(X, Y, 5)
	bhat,shat = @time jumpMlogitMLE(Y,X)
end