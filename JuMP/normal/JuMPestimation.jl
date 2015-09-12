#  This script runs estimation of the Classical Linear Regression Model using Julia for Mathematical Programming (JuMP)
#  The script has been adapted from code posted on Github at https://github.com/JuliaOpt/JuMP.jl/blob/master/examples/mle.jl
using JuMP, NLopt, Ipopt

include("datagen.jl")
include("jumpMLE.jl")
# include("jumpOLS.jl")

# evaluate the functions referenced above in the -include- statements. -@time- is equivalent to tic/toc in Matlab
@time X,Y,bAns,sigAns,n = datagen() # specify function outputs on the left hand side of the equals sign
@time bOpt,sOpt,hOpt,seOpt = jumpMLE(Y,X,[bAns,sigAns])
# @time jumpOLS()