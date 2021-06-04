function av = average_quantity(quantity, Nf, t, n, X)

quantity_lower = quantity ;
av = zeros(Nf,t);
for f = 1:Nf
    for j = 1:t
        collector =0;
        for i = 1:n
            collector = collector + quantity_lower{i}(f,j);
        end
        av(f,j) = av(f,j) + collector/n;
    end
end

av = sum(sum(av));
return 