% NZ's age-wage profile based on Papps (2010).
clear all;
b1 = 0.046;
b2 = -0.049;
gens = 45;
% The number of working ages is 45 from 20 to 64.
% The number of retired ages is 35 from 65 to 99.
x = zeros(gens,2);
for i = 1:gens;
    x(i,1)=i+19;
    x(i,2)=b1*x(i,1)+b2*(x(i,1))^2/100;
end
age = x(:,1);
wage = x(:,2);
wage_normalised = x(:,2)./mean(x(:,2));
% plot(age,wage_normalised)
% xlabel("Age (-20)")
% ylabel("Productivity")
fprintf('Average wage normalised to one：%.3f \n', mean(wage_normalised))
% x(:,2)

% Convert the annual data to five-year frequency.
nw = 9;
efage = zeros(nw,1);
realage = zeros(nw,1);
i = 0;
while i < nw
    i = i + 1;
    efage(i) = mean(wage((i-1)*5+1:i*5));
    realage(i) = (i-1)*5 + 20;
end
efage = efage/mean(efage);
fprintf("efage (average)：%.3f \n", mean(efage));
plot(realage, efage, '-o');
title("Age profile of individual labour productivity")
xlabel("Real age")
ylabel("Normalised productivity")
df_efage = mat2dataset(efage);
export(df_efage, 'File', './efage.csv', "WriteVarNames", false)
