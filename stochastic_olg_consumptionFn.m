% function c=OLGModel5_ConsumptionFn(h,aprime,a,agej,Jr,r,pension,tau,kappa_j,alpha,delta,A,eta1,eta2)
function c = stochastic_olg_consumptionFn(h,aprime,a,z, agej,Jr,r,pension,kappa_j,alpha,delta,A, tau_c, tau_y)
% Note: these lines are essentially just a copy of the relevant part of the return fn

KdivL=((r+delta)/(alpha*A))^(1/(alpha-1));
w=A*(1-alpha)*(KdivL^alpha); % wage rate (per effective labour unit)

% Flat income tax
Income=w*kappa_j*z*h+r*a; % Income is labor income and capital income
% Income=w*kappa_j*h+r*a; % Income is labor income and capital income
% Progressive income tax
% IncomeTax=eta1+eta2*log(Income)*Income;
IncomeTax = tau_y * Income;

% if agej<Jr
c = 0
if agej<=Jr
%    c=(1+r)*a+(1-tau)*w*kappa_j*h-IncomeTax-aprime; % Working age
    c=(1+r)*a + w*kappa_j*z*h - IncomeTax -aprime; % Working age
else
    c=(1+r)*a+pension-IncomeTax - aprime; % Retired
end

% 自分で修正。消費税の分を考慮する。
c = c / (1+tau_c);

end
