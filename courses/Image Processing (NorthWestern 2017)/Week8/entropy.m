function result = entropy(probs)
    result = -sum(probs .* arrayfun(@log2m, probs));
end