#  This script runs estimation of the Classical Linear Regression Model using Julia for Mathematical Programming (JuMP)
#  The script has been adapted from code posted on Github at https://github.com/JuliaOpt/JuMP.jl/blob/master/examples/mle.jl
using DataFrames
using StatsBase
using Distributions
using Debug
using JuMP
using Ipopt

include("simulation2.jl")
include("pclogit.jl")
include("ewage.jl")
include("benefits.jl")
include("estimation_simpleFD.jl")
include("jumpMlogitMLE.jl")

# evaluate the functions referenced above in the -include- statements. -@time- is equivalent to tic/toc in Matlab
@time b_ans,bs,bm,Y,LY,FY,lnEarn,obsloc,exper0,exper,T,N,nloc,time,p1,p2,p3,p4,p5,p6,fv = simulation2() # specify function outputs on the left hand side of the equals sign

@time estimation_simpleFD(b_ans,bs,bm,Y,LY,FY,lnEarn,obsloc,exper0,exper,T,N,nloc,time,p1,p2,p3,p4,p5,p6,fv)

# tabulate the choice variable to check cell sizes
# tabs = zeros(J)
# for j=1:J
	# tabs[j]=100*(sum(Y.==j)/length(Y))
# end
# println("frequency of choices:", tabs)

# # optimize
# @time bxOpt,bzOpt,hOpt = jumpMlogitMLE()

# # compare answers
# bVec = [vec(bxOpt), vec(bzOpt)]
# hessie = -full(hOpt)
# bxOpt-broadcast(-,bxAns,bxAns[:,end])
# bzAns-bzOpt
