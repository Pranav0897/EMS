function [ok] = myfmincon(nbus, P, Q, Pij, Qij, linedatas, ybus_real, ybus_imag)

errorlb = -inf*ones(length(P)+length(Q)+length(Pij)+length(Qij),1);
errorub = inf*ones(length(P)+length(Q)+length(Pij)+length(Qij),1);
V_estub = 1.1*ones(length(P),1);
V_estlb = 0.9*ones(length(P),1);
phi_estub = (pi)*ones(length(P),1);
phi_estlb = -(pi)*ones(length(P),1);
phi_estlb(1) = 0;
phi_estub(1) = 0;

P = cell2mat(P); 
Q = cell2mat(Q); 
Pij = cell2mat(Pij); 
Qij = cell2mat(Qij); 
linedatas = vertcat(linedatas{:});
linedatas = cell2mat(linedatas);
ybus_real = vertcat(ybus_real{:});
ybus_real = cell2mat(ybus_real);
ybus_imag = vertcat(ybus_imag{:});
ybus_imag = cell2mat(ybus_imag);

%P = str2double(P); 
%Q = str2double(Q); 
%Pij = str2double(Pij); 
%Qij = str2double(Qij); 
%linedatas = str2double(linedatas);
%ybus_real = str2double(ybus_real);
%ybus_imag = str2double(ybus_imag);

%P = cellfun(@(x) double(x), P); 
%Q = cellfun(@(x) double(x), Q); 
%Pij = cellfun(@(x) double(x), Pij); 
%Qij = cellfun(@(x) double(x), Qij); 
%linedatas = cellfun(@(x) double(x), linedatas); 
%ybus_real = cellfun(@(x) double(x), ybus_real);
%ybus_imag = cellfun(@(x) double(x), ybus_imag);

ybus = ybus_real + 1i*ybus_imag;

x0 = [zeros(length(P)+length(Q)+length(Pij)+length(Qij),1);ones(nbus,1);zeros(nbus,1)];
x0 = double(x0);
options = optimset('Display','on','algorithm','sqp');
%tic
[x_old val] = fmincon(@(x) objse(x,P,Q,Pij,Qij),x0,[],[],[],[],[errorlb',V_estlb',phi_estlb'],[errorub',V_estub',phi_estub'],@(x) nlconse(x,P,Q,Pij,Qij,linedatas,ybus),options);
%toc
Vse = x_old(length(P)+length(Q)+length(Pij)+length(Qij)+1:length(P)+length(Q)+length(Pij)+length(Qij)+nbus);
phise = x_old(length(P)+length(Q)+length(Pij)+length(Qij)+nbus+1:length(P)+length(Q)+length(Pij)+length(Qij)+2*nbus);
