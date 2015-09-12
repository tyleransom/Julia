#  This script runs estimation of the Classical Linear Regression Model using Julia for Mathematical Programming (JuMP)
#  The script has been adapted from code posted on Github at https://github.com/JuliaOpt/JuMP.jl/blob/master/examples/mle.jl
using DataFrames
using JuMP
using NLopt
using Ipopt

include("datagenMlogit.jl")
include("jumpMlogitMLE.jl")

# evaluate the functions referenced above in the -include- statements. -@time- is equivalent to tic/toc in Matlab
@time X,Y,Z,bxAns,bzAns,n,J,K1,K2,baseAlt = datagenMlogit() # specify function outputs on the left hand side of the equals sign

# tabulate the choice variable to check cell sizes
tabs = zeros(J)
for j=1:J
	tabs[j]=100*(sum(Y.==j)/length(Y))
end
println("frequency of choices:", tabs)

# optimize
@time bxOpt,bzOpt,hOpt = jumpMlogitMLE()

# compare answers
bVec = [vec(bxOpt), vec(bzOpt)]
hessie = -full(hOpt)
bxOpt-broadcast(-,bxAns,bxAns[:,end])
bzAns-bzOpt
