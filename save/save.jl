using Debug,Distributions,MAT

# file = matopen("carbig.mat")
# read(file, "varname") # note that this does NOT introduce a variable ``varname`` into scope
# close(file)

file = matopen("carbig.mat")
varnames = names(file)
close(file)

vars = matread("carbig.mat")
Weight       = vars["Weight"]
Acceleration = vars["Acceleration"]
Mfg          = vars["Mfg"]
cyl4         = vars["cyl4"]
Origin       = vars["Origin"]
when         = vars["when"]
Displacement = vars["Displacement"]
MPG          = vars["MPG"]
Model        = vars["Model"]
Cylinders    = vars["Cylinders"]
org          = vars["org"]
Model_Year   = vars["Model_Year"]
Horsepower   = vars["Horsepower"]

X=rand(50000,5)

matwrite("tester.mat", {
    "X" => X,
    "Acceleration" => Acceleration,
    "Horsepower" => Horsepower
})
