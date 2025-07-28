%% Pearson VII kernel
function K = PUK_kernel(U, V)
    sigma=1.0;
    omega=1.0;
    D = pdist2(U, V, 'squaredeuclidean');
    factor = 4 * sqrt(2^(1/omega) - 1);
    K = (1 + (D * factor / sigma^2)) .^ omega;
    K = 1 ./ K;
end