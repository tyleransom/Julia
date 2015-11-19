function olsobjfun(beta)
	err = sum((Y-X*beta).^2)
	return err
end