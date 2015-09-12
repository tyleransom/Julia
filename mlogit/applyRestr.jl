function applyRestr(restrMat,b,H=[])
#APPLYRESTR applies restrictions to a model
#   B = APPLYRESTR(RESTRMAT,B) implements restrictions on the vector B 
#   of parameters according to the specifications found in RESTRMAT. 
#   Two types of restrictions are supported:
#
#     Type 1  Restricting one parameter ("parmA") to equal a fixed value
#     Type 2  Restricting one parameter, parmA, to equal another ("parmB"),
#             potentially multiplied by some real number q and addd to
#             some constant m, e.g. parmA = m + q*parmB.
#   
#   RESTRMAT follows a very specific format. It is an R-by-5 matrix, 
#   where R is the number of restrictions. The role of each of the four 
#   columns is as follows
#
# 	  Column 1  The index of parmA
# 	  Column 2  The index of parmB (zero if type 1 restriction)
# 	  Column 3  Binary vector where 0 indciates a type 1 restriction (parmA
#               set equal to fixed value) and 1 indicates a type 2 
#               restriction (parmA set equal to parmB)
# 	  Column 4  If a type 1 restriction, 0. If a type 2 restriction, any 
#               real number q such that parmA = q*parmB.
# 	  Column 5  If a type 1 restriction, the fixed value. If a type 2
#               restriction, any real number m such that parmA = m+q*parmB.
#
#   APPLYRESTR does not allow for any combination of restrictions. If 
#   two parameters are to be restricted to the same fixed value, they
#   should both be type 1 restrictions rather than a type 1 restriction
#   and a type 2 restriction
#   
#   The same parameter cannot appear in Column 1 of RESTRMAT twice. For 
#   restrictions involving multiple parameters, e.g. b(1) = b(2) = b(3),
#   create two restrictions: 1) b(1) = b(3); and 2) b(2) = b(3).
#
#   Note that RESTRMAT must be sorted in ascending order based on 
#   column 1. This is especially important for hessian maatrix
#   manipulation, explained below.
#
#   [B,INVH] = APPLYRESTR(RESTRMAT,B,H) takes as input a hessian matrix H
#   (typically from an optimization routine) and returns an inverted
#   hessian INVH where restrictions have been applied. A type 1 restriction
#   results in a row and a column of zeroes in the hessian at the index for
#   that paramter. A type 2 restriction duplicates
#   the rows and columns for parmA and parmB. This implies that the 
#   covariance between parmA and parmB is set equal to their variance.
#   Moreover, the covariance of parmA and parmB with other parameters is
#   restricted to be equal.
#
# 
# Copyright 2014 Jared Ashworth and Tyler Ransom, Duke University
# Special thanks to Vladi Slanchev
# Revision History: 
#   July 3, 2013
#     Created
#   November 13, 2013
#     Error message if restrMat is empty or wrong size
#   November 15, 2013
#     Remove empty rows
#   July 19, 2014
#     Published
#----------------------------------------------------------------------------
assert(~isempty(restrMat),"Empty restriction matrix");
assert(size(restrMat,2)==5,"Restriction matrix requires 5 columns");

restrMat = restrMat[restrMat[:,1]>0,:]; # Remove empty rows
assert(~isempty(restrMat),"Restriction matrix has no positive indices");
assert(max(restrMat[:,1])<=size(b,1),"Restriction matrix has indices beyond size of parameter vector");


bRestr=b;
restrMat=sortrows(restrMat); # need to specify the column that the sorting is based on ???
R = size(restrMat,1);
for r=1:R
	if restrMat[r,3]==0
		bRestr[restrMat[r,1]]=restrMat[r,5];
	elseif restrMat[r,3]==1
		bRestr[restrMat[r,1]]=restrMat[r,5]+restrMat[r,4]*bRestr[restrMat[r,2]];
	end
end

for r=R:-1:1
	H[restrMat[r,1],:]=[];
	H[:,restrMat[r,1]]=[];
end
invH = full(H)\eye(size(H,1));
for r=1:R
	invH= [ invH[1:restrMat[r,1]-1,:]; zeros(1,size[invH,1]); invH[restrMat[r,1]:end,:]];
	invH= [ invH[:,1:restrMat[r,1]-1]  zeros(size[invH,1],1)  invH[:,restrMat[r,1]:end]];
end
for r=1:R
	if restrMat[r,3]==1
		invH[restrMat[r,1],:]=restrMat[r,4]*invH[restrMat[r,2],:];
		invH[:,restrMat[r,1]]=restrMat[r,4]*invH[:,restrMat[r,2]];
	end
end
invHrestr = invH;

return bRestr,invHrestr
end