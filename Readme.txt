This is a repository designed to introduce basic Julia applications for econometrics. The folders included are as follows:
- dataframes: Basic introduction to Julia's dataframes capabilities
- DDC: An attempt at solving a dynamic discrete choice model
- JuMP: Various applications of Julia's JuMP package (which is a modelling language that interfaces with a variety of industry-grade optimizers). Subfolders are as follows:
	* mlogit: classic multinomial logit with alternative-specific parameters, estimated with and without constraints
	* normal: maximum likelihood estimation of the classical normal linear model
	* xtlogit: a (currently unsuccessful) attempt at a binary logit with random effects
	* xtlogitNoHet: binary logit with no random effects
	* xtlogitSimple: another attempt at a binary logit with random effects
- loopVsMatmul: a speed test for matrix multiplication in loop versus vectorized form
- mlogit: Vestigial version of JuMP/mlogit/, before I discovered JuMP
- ols: basic OLS in Julia
