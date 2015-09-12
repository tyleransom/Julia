#  This script runs estimation of the Classical Linear Regression Model using Julia for Mathematical Programming (JuMP)
#  The script has been adapted from code posted on Github at https://github.com/JuliaOpt/JuMP.jl/blob/master/examples/mle.jl
using DataFrames
using Distributions
using GLM
using JuMP
using Ipopt
using Regression

include("analysis.jl")
include("jumpMLE.jl")
include("jumpMlogitMLE.jl")

# evaluate the functions referenced above in the -include- statements. -@time- is equivalent to tic/toc in Matlab
@time analysis()