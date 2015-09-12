function ewage(nloc,j,e,b);
# This function computes expected wages given state variables
if length(j)==1
	locer = zeros(1,nloc-1);
	if j>nloc
		w = zeros(size(e));
	else
		if j>1
			locer[j-1] = 1;
		end
		w = [ones(size(e)) kron(ones(size(e)),locer) e e.^2]*b;
	end
else
	locer = zeros(length(e),nloc-1);
	for k=2:nloc
		locer[:,k-1] = j.==k;
	end
	if minimum(j)>nloc
		w = zeros(size(e));
	else
		w = [ones(size(e)) locer e e.^2]*b;
	end
end

return w
end