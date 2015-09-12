#  This script times loop multiplication vs. matrix multiplication in Julia

include("datagen.jl")
include("matmul.jl")
include("loopmul.jl")


@time X,beta   = datagen(convert(Int64,1e5))
@time Ymatmul  = matmul(X,beta)
@time Yloopmul = loopmul(X,beta)