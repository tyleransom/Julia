function applyRestrGrad(restrMat,grad,hess)
#APPLYRESTRGRAD applies restrictions to the gradient of an objective function
#   G = APPLYRESTR(RESTRMAT,GRAD) implements restrictions on the gradient
#   vector GRAD of parameters according to the specifications found in
#   RESTRMAT. See APPLYRESTR for more details on constructing the
#   restriction matrix RESTRMAT.

# Copyright 2014 Jared Ashworth and Tyler Ransom, Duke University
# Special thanks to Vladi Slanchev
# Revision History: 
#   July 3, 2013
#     Created
#   July 9, 2014
#     Generalize gradient to fit any objective function
#   July 19, 2014
#     Published
#----------------------------------------------------------------------------
restrMat = restrMat[restrMat[:,1]>0,:]; # Remove empty rows

gRestr=grad;
hRestr=hess;
restrMat=sortrows(restrMat); # need to specify the column that the sorting is based on ???
R = size(restrMat,1);
if R>0
	# gradient
	for r=1:R
        i = restrMat[r,1];
        h = restrMat[r,2];
        gRestr[i]=0;
		if restrMat[r,3]==1
			gRestr[h]=gRestr[h]+restrMat[r,4]*grad[i];
		end
	end
	
	# hessian
	for r=1:R
        i = restrMat(r,1);
        h = restrMat(r,2);
        hRestr[i,:]=zeros(1,size(hess,2));
        hRestr[:,i]=zeros(size(hess,1),1);
	end
end	

return gRestr,hRestr
end