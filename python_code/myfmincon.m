function [x_old] = myfmincon(nbus, P, Q, Pij, Qij, linedatas, ybus)

errorlb = -inf*ones(length(P)+length(Q)+length(Pij)+length(Qij),1);
errorub = inf*ones(length(P)+length(Q)+length(Pij)+length(Qij),1);
V_estub = 1.1*ones(length(P),1);
V_estlb = 0.9*ones(length(P),1);
phi_estub = (pi)*ones(length(P),1);
phi_estlb = -(pi)*ones(length(P),1);
phi_estlb(1) = 0;
phi_estub(1) = 0;

%P = cell2mat(P); 
%Q = cell2mat(Q); 
%Pij = cell2mat(Pij); 
%Qij = cell2mat(Qij); 
%linedatas = cell2mat(linedatas); 
%ybus = cell2mat(ybus);

P = cellfun(@(x) double(x), P); 
Q = cellfun(@(x) double(x), Q); 
Pij = cellfun(@(x) double(x), Pij); 
Qij = cellfun(@(x) double(x), Qij); 
linedatas = cellfun(@(x) double(x), linedatas); 
ybus = cellfun(@(x) double(x), ybus);

x0 = [zeros(length(P)+length(Q)+length(Pij)+length(Qij),1);ones(nbus,1);zeros(nbus,1)];
x0 = double(x0);
options = optimset('Display','on','algorithm','sqp');
%tic
[x_old val] = fmincon(@(x) objse(x,P,Q,Pij,Qij),x0,[],[],[],[],[errorlb',V_estlb',phi_estlb'],[errorub',V_estub',phi_estub'],@(x) nlconse(x,P,Q,Pij,Qij,linedatas,ybus),options);
%toc
%Vse = x_old(length(P)+length(Q)+length(Pij)+length(Qij)+1:length(P)+length(Q)+length(Pij)+length(Qij)+nbus);
%phise = x_old(length(P)+length(Q)+length(Pij)+length(Qij)+nbus+1:length(P)+length(Q)+length(Pij)+length(Qij)+2*nbus);
