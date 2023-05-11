%% This file solves the OLG model for the bench mark scenario.
%% Bench mark scenario: Public pension system.
clear all;
vfoptions.exoticpreferences = 'QuasiHyperbolic';
vfoptions.quasi_hyperbolic = 'Naive';
% vfoptions.quasi_hyperbolic = 'Sophisticated';
Params.beta0 = 0.9;
Params.J=15; % Number of periods. The length of one period is 5 years. 

% Grid sizes to use
n_d=101; % For labour choice (fraction of time worked).
n_a=301; % For asset holdings.
n_z = 2; % For employment status.
N_j=Params.J; % Number of periods in finite horizon.

figure_c=0; % Use a counter for the figures. 

%% Parameters
% Deterministic growth
Params.g = (1+0.0146)^5 - 1;

% Discount rate
Params.beta = 1.021;
% Preferences
Params.eta = 1.11;
Params.phi = 0.5;
Params.kappa = 15.1; 

Params.A=1; % Aggregate TFP. Not actually used anywhere.
% Production function
Params.alpha = 0.42; % Share of capital.
delta_q = 0.0116; % Quarterly depreciation rate of capital.
delta_a = 1 - (1-delta_q)^4; % Annual depreciation rate of capital.
Params.delta = 1 - (1-delta_a)^5; % 5-year depreciation rate of capital.
% Warm-glow of bequest
% Params.warmglowparam1=1;
% Params.warmglowparam2=2;

% Demographics
Params.Jr = 9; % Retirement age in the model, which is 66 in real life.
Params.agej=(1:1:Params.J)'; % Current 'j' age, so can check when you retire.
% Population growth rate
Params.n = table2array(readtable('./n.csv', 'Encoding', 'UTF-8'));
% Params.n = 0.011125; % This is same as above.
% Params.n = 0.5/100; % This is the lower population growth rate.

% Conditional survival probabilities: sj is the probability of surviving to be age j+1, given alive at age j
Params.sj = table2array(readtable('./sp.csv', 'Encoding', 'UTF-8'));

% Labor efficiency units depend on age
for_kappa_j = table2array(readtable('./efage.csv', 'Encoding', 'UTF-8'))';
Params.kappa_j = [for_kappa_j, zeros(1,(N_j-Params.Jr))];
% Taxes
Params.tau_c = 0.15; % Consumption tax rate

% Government spending
Params.GdivYtarget = 0.181; % Government spending as a fraction of GDP (this is essentially just used as a target to define a general equilibrium condition)
Params.BdivYtarget = 0.419/5; % Outstanding government bonds as a fraction of 5-year GDP. 
Params.replacement = 0.40; % Replacement ratio of transfer to labour income.

%% Some initial values/guesses for variables that will be determined in general eqm
Params.pension=0.0795; % Initial guess for pension payment (this will be determined in general eqm)
Params.r=0.227; % Initial guess for interest rate
Params.G=0.035; % Initial guess for government expenditure
Params.tau_y = 0.5; % Initial guess for income tax rate
Params.kshare = 0.9; % Initial guess for share of capital in assets. 自分で追加。

%% Grids
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.
% Note: Implicitly, we are imposing borrowing constraint a'>=0
z_grid = [1;0]; % The first entry is employment and the second is unemployment.
pi_z = [0.9715, 0.0285; 0.5, 0.5]; % p_ee=0.9715, p_eu=0.0285, p_ue=p_uu=0.5.
% Grid for labour choice
h_grid=linspace(0,1,n_d)';
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function 
Params.growthdiscountfactor=(1+Params.g)^(1-Params.eta);
DiscountFactorParamNames={'beta','sj','growthdiscountfactor', 'beta0'};

ReturnFn=@(h,aprime,a,z, agej,r,A,delta,alpha, eta, kappa, phi,Jr,pension,tau_y, tau_c, kappa_j)...
    stochastic_olg_returnFn(h,aprime,a,z, agej,r,A,delta,alpha, eta, kappa, phi,Jr,pension,tau_y, tau_c, kappa_j);


%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% Plot the value function calculated. 自分でLifeCycleModel4を参考に加筆。
figure_c = figure_c + 1;
subplot(2,1,1)
surf(a_grid*ones(1,Params.J),ones(n_a,1)*(21:5:(21+5*(Params.J-1))),reshape(V(:,1,:),[n_a,Params.J])) % The reshape is needed to get rid of z
title('The Employed.')
xlabel('Assets (a)')
ylabel('Real age')
subplot(2,1,2)
surf(a_grid*ones(1,Params.J),ones(n_a,1)*(21:5:(21+5*(Params.J-1))),reshape(V(:,2,:),[n_a,Params.J])) % The reshape is needed to get rid of z
title('The Unemployed')
xlabel('Assets (a)')
ylabel('Real age')
sgtitle('Value Function')

%% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros(n_a,n_z, 'gpuArray'); % n_a by n_z.
jequaloneDist(1,1:n_z)=[1.0, 0.0]; % Everyone is born with zero assets and will be employed.

%% Agents age distribution
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1)/(1+Params.n);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one

AgeWeightsParamNames={'mewj'}; % Apply different weights to different 'ages' due to survival rates and population growth rates.

%% Test
disp('Test StationaryDist')
simoptions=struct(); % Just use the defaults
tic;
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
toc
%% General eqm variables
GEPriceParamNames={'r','pension','G','tau_y', 'kshare'};

%% Set up the General Equilibrium conditions
% Stationary Distribution Aggregates (important that ordering of Names and Functions is the same)
FnsToEvaluate.H = @(h,aprime,a,z) h; % Aggregate labour supply (in efficiency units)
FnsToEvaluate.L = @(h,aprime,a,z, kappa_j) kappa_j*h; % Aggregate labour supply (in efficiency units)
FnsToEvaluate.Asset = @(h,aprime,a,z) a; % Aggregate assets
FnsToEvaluate.Aprime = @(h,aprime,a,z) aprime; % Aggregate assets in the next period
FnsToEvaluate.PensionSpending = @(h,aprime,a,z, pension,agej,Jr) (agej>=Jr)*pension; % Total spending on pensions
FnsToEvaluate.AccidentalBeqLeft = @(h,aprime,a,z, sj) aprime*(1-sj); % Accidental bequests left by people who die
FnsToEvaluate.IncomeTaxRevenue = @(h,aprime, a, z, kappa_j,r,delta,alpha,A, tau_y) stochastic_olg_incometaxFn(h,aprime, a,z, kappa_j,r,delta,alpha,A, tau_y);
FnsToEvaluate.Consumption=@(h,aprime,a,z, agej,Jr,r,pension,kappa_j,alpha,delta,A, tau_c, tau_y) stochastic_olg_consumptionFn(h,aprime,a,z, agej,Jr,r,pension,kappa_j,alpha,delta,A, tau_c, tau_y);

% General Equilibrium conditions (these should evaluate to zero in general equilbrium)
GeneralEqmEqns.capitalmarket = @(r,Asset, kshare, L,alpha,delta,A) r-alpha*A*((kshare*Asset)^(alpha-1))*(L^(1-alpha)); % interest rate equals marginal product of capital net of depreciation
GeneralEqmEqns.pensions = @(PensionSpending, replacement, alpha, A, r, delta, L)...
    PensionSpending - replacement*((1-alpha)*A*((r+delta)/(alpha*A))^(alpha/(alpha-1)))*L;
GeneralEqmEqns.govbudget = @(PensionSpending,IncomeTaxRevenue, AccidentalBeqLeft, tau_c,Consumption, r, kshare, Aprime, Asset, G) PensionSpending-IncomeTaxRevenue - AccidentalBeqLeft - tau_c*Consumption + G - (1-kshare)*Aprime + (1+r)*(1-kshare)*Asset;
GeneralEqmEqns.Gtarget = @(G,GdivYtarget,A,Asset,kshare,L,alpha) G-GdivYtarget*(A*(kshare*Asset)^(alpha)*(L^(1-alpha))); % G is equal to the target, GdivYtarget*Y
GeneralEqmEqns.Identity = @(A,Asset,kshare,Aprime, alpha,L,Consumption, g, n, delta, G, B) A*(kshare*Asset)^(alpha)*L^(1-alpha) - G - Consumption - kshare*(Aprime -(1-delta)*Asset);

%% Test
disp('Test AggVars')
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid);

%% Solve for the General Equilibrium
heteroagentoptions.verbose=1;
p_eqm=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightsParamNames,n_d, n_a, n_z, N_j, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
% p_eqm contains the general equilibrium parameter values
% Put this into Params so we can calculate things about the initial equilibrium
Params.r=p_eqm.r;
Params.pension=p_eqm.pension;
Params.tau_y = p_eqm.tau_y;
Params.G=p_eqm.G;
Params.kshare =p_eqm.kshare;

% Calculate a few things related to the general equilibrium.
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% Can just use the same FnsToEvaluate as before.
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid);

%% Plot the life cycle profiles of capital and labour for the inital and final eqm.
figure_c=figure_c+1;
figure(figure_c)
real_ages = (21:5:(21+5*(Params.J-1)));
real_working_ages =(21:5:(21+5*(Params.Jr-1)));
subplot(2,2,1); plot(real_ages,AgeConditionalStats.Asset.Mean)
title('Assets')
xlabel('Real age')
subplot(2,2,2); plot(real_working_ages,AgeConditionalStats.H.Mean(1:Params.Jr))
title('Proportion of time spent at work')
xlabel('Real age')
subplot(2,2,3); plot(real_ages,AgeConditionalStats.Consumption.Mean)
title('Consumption')
xlabel('Real age')
subplot(2,2,4); plot(real_ages,(AgeConditionalStats.IncomeTaxRevenue.Mean/Params.tau_y - AgeConditionalStats.Consumption.Mean))
title('Saving rate')
xlabel('Real age')
sgtitle('Life Cycle Profiles')
writematrix(AgeConditionalStats.Asset.Mean,'./Pension/LifeCycleProfiles_Asset_Pension.txt'); 
writematrix(AgeConditionalStats.H.Mean,'./Pension/LifeCycleProfiles_H_Pension.txt'); 
writematrix(AgeConditionalStats.Consumption.Mean,'./Pension/LifeCycleProfiles_Consumption_Pension.txt'); 
writematrix((AgeConditionalStats.IncomeTaxRevenue.Mean/Params.tau_y - AgeConditionalStats.Consumption.Mean), './Pension/LifeCycleProfiles_SavingRate_Pension.txt'); 

figure_c=figure_c+1;
figure(figure_c);
retired_ages = real_ages(Params.Jr+1:Params.J);
retired_pensions = ones(1,(Params.J-Params.Jr))*Params.pension;
all_age_pensions = [zeros(1,Params.Jr) retired_pensions];
plot(retired_ages, retired_pensions./AgeConditionalStats.IncomeTaxRevenue.Mean((Params.Jr+1):Params.J))
title('Average ratio of social benefit transer to income tax per person')
xlabel('Real age')
writematrix(retired_pensions./AgeConditionalStats.IncomeTaxRevenue.Mean((Params.Jr+1):Params.J),'./Pension/LifeCycleProfiles_pensiontoincometax_Pension.txt'); 

%% Calculate some aggregates and print findings about them
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid);
K = AggVars.Asset.Mean*Params.kshare;

% GDP
Y=Params.A*(K^Params.alpha)*(AggVars.L.Mean^(1-Params.alpha));

% wage (note that this is calculation is only valid because we have Cobb-Douglas production function and are looking at a stationary general equilibrium)
KdivL=((Params.r+Params.delta)/(Params.alpha*Params.A))^(1/(Params.alpha-1));
w=Params.A*(1-Params.alpha)*(KdivL^Params.alpha); % wage rate (per effective labour unit)

fprintf('Following are some aggregates of the model economy: \n')
fprintf('Output: Y=%8.2f \n',Y)
fprintf('Capital-Output ratio: K/Y=%8.2f \n',K/Y)
fprintf('Consumption-Output ratio: C/Y=%8.2f \n',AggVars.Consumption.Mean/Y)
fprintf('Average labor productivity: Y/H=%8.2f \n', Y/AggVars.H.Mean)
fprintf('Government-to-Output ratio: G/Y=%8.2f \n', Params.G/Y)
fprintf('Wage: w=%8.2f \n',w)
fprintf('Income tax rate: tau_y=%8.3f \n', Params.tau_y)
fprintf('Real interest rate (percent): r=%8.2f \n', ((1+Params.r)^(1/5)-1)*100)
fprintf('Aggregate consumption: C=%8.2f \n', AggVars.Consumption.Mean)
fprintf('Aggregate capital: K=%8.2f \n', K)

writematrix(Params.tau_y, './Pension/Income_tax_rate_Pension.txt')
writematrix(Params.tau_c, './Pension/Consumption_tax_rate_Pension.txt')
writematrix(Params.kshare, './Pension/kshare_Pension.txt')
writematrix(AggVars.Asset.Mean, './Pension/Average_Asset_Pension.txt')
writematrix(K, './Pension/Average_K_Pension.txt')
writematrix(Y, './Pension/Average_Y_pension.txt')
writematrix(AggVars.Consumption.Mean, './Pension/Average_C_pension.txt')

%% Look at some further model outputs
LorenzCurve=EvalFnOnAgentDist_LorenzCurve_FHorz_Case1(StationaryDist,Policy, FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,[],simoptions);
writematrix(LorenzCurve.Asset, './Pension/Lorenz_Asset_Pension.txt');
writematrix(LorenzCurve.Consumption, './Pension/Lorenz_Consumption_Pension.txt');

% Returns a Lorenz Curve 100-by-1 that contains all of the quantiles from 1
% to 100. Unless the simoptions.npoints is set which case it will be npoints-by-1.
figure_c=figure_c+1;
figure(figure_c)
subplot(2,1,1); plot(LorenzCurve.Asset)
title('Lorenz curve of assets')
subplot(2,1,2); plot(LorenzCurve.Consumption)
title('Lorenz curve of consumption')
% % Once you have a Lorenz curve you can calculate the Gini coefficient using
Gini_Asset=Gini_from_LorenzCurve(LorenzCurve.Asset); 
writematrix(Gini_Asset,'./Pension/Gini_Asset_Pension.txt'); 
Gini_Consumption = Gini_from_LorenzCurve(LorenzCurve.Consumption); 
writematrix(Gini_Consumption,'./Pension/Gini_Consumption_Pension.txt'); 

% Calculate the average time spent at work among labour forces.
HtimesPop = AgeConditionalStats.H.Mean.*Params.mewj;
HtimesWorkingPop = HtimesPop(1:Params.Jr);
AvWorkTime = sum(HtimesWorkingPop)/sum(Params.mewj(1:Params.Jr));
fprintf('Average time spent at work (among labour forces) should be 0.263 and is %8.4f \n', AvWorkTime);
writematrix(AvWorkTime,'./Pension/Average_time_at_work_Pension.txt'); 
fprintf('Pension per person: =%8.4f \n', Params.pension);
writematrix(Params.pension,'./Pension/Pension_perperson_Pension.txt'); 
