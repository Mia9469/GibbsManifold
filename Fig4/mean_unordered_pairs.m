function value = mean_unordered_pairs(similarity)
% Average each distinct trial pair once; zero similarity is a valid result.
assert(ismatrix(similarity) && size(similarity, 1) == size(similarity, 2) ...
    && size(similarity, 1) >= 2, 'Expected a square pair-similarity matrix');
mask = triu(true(size(similarity)), 1);
assert(all(isfinite(similarity(mask))), 'Non-finite trial-pair similarity');
value = mean(similarity(mask));
end
