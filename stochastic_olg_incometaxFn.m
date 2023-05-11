% function IncomeTax=OLGModel5_ProgressiveIncomeTaxFn(h,aprime,a,eta1,eta2,kappa_j,r,delta,alpha,A)
function IncomeTax=stochastic_olg_incometaxFn(h,aprime, a,z, kappa_j,r,delta,alpha,A, tau_y)

KdivL=((r+delta)/(alpha*A))^(1/(alpha-1));
w=A*(1-alpha)*(KdivL^alpha); % wage rate (per effective labour unit)

% Progressive income tax
% Income=w*kappa_j*h+r*a; % Income is labor income and capital income
Income=w*kappa_j*z*h+r*a; % Income is labor income and capital income
IncomeTax=0;
if Income>0
%    IncomeTax=eta1+eta2*log(Income)*Income;
   IncomeTax=tau_y * Income;
end
% Note: Have made pensions exempt from income tax.

end