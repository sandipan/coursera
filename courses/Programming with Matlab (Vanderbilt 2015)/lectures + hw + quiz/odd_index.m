function Modd = odd_index(M)
	Modd = M(mod([1:size(M,1)], 2) == 1, mod([1:size(M,2)], 2) == 1); 
end