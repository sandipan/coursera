function p = prime_pairs(n)
    p = -1;
    ps = primes(200000);
    pps = intersect(ps, ps-n);
    if (numel(pps) > 0)
        p = pps(1);
        if p >= 100000
            p = -1;
        end
    end
end
