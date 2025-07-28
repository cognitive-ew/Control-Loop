% Pearson VII kernel
%
% Ustun, Melssen, and Buydens, "Facilitating the application of Support Vector
% Regression by using a Universal Pearson VII Function Based Kernel," Chemometrics and
% Intelligent Laboratory Systems, Vol. 81, No. 1, 2006.
% DOI: 10.1016/j.chemolab.2005.09.003.
					      
function K = PUK_kernel(U, V)
    sigma=1.0;
    omega=1.0;
    D = pdist2(U, V, 'squaredeuclidean');
    factor = 4 * sqrt(2^(1/omega) - 1);
    K = (1 + (D * factor / sigma^2)) .^ omega;
    K = 1 ./ K;
end
