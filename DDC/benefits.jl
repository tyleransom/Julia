function benefits(nloc,j,e);
# This function computes unemployment benefits given state variables
	if j>nloc
		b = ones(size(e));
	else
		b = zeros(size(e));
	end
	return b
end