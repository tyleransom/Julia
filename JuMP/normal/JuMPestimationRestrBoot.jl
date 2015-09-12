#  This script runs estimation of the Classical Linear Regression Model using Julia for Mathematical Programming (JuMP)
#  The script has been adapted from code posted on Github at https://github.com/JuliaOpt/JuMP.jl/blob/master/examples/mle.jl
using JuMP
using NLopt
using Ipopt
using Distributions
using Debug

include("datagenBoot.jl")
include("jumpMLErestrBoot.jl")
include("bootstrapper.jl")

# evaluate the functions referenced above in the -include- statements. -@time- is equivalent to tic/toc in Matlab
@time x,y,bAns,sigAns,n = datagenBoot() # specify function outputs on the left hand side of the equals sign
@time bOptBoot,seOptBoot,bootEst,fvalEst = bootstrapper(100)
