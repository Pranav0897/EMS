function [Vse phise]=state_estimate(path_to_file, P,Q,Pij,Qij)
%% MATPOWER to MYFORMAT
% [busdatas, linedatas, gencost] = myformat(ext2int(casefile));
casefile=load(path_to_file);
casefile=casefile.a_dict;
[busdatas, linedatas, gencost] = myformat(casefile);
nbus = max(max(linedatas(:,1)),max(linedatas(:,2)));

%gencost(:,2)=0;
%% Ybus calculation

[ybus A] = ybus_incidence(linedatas,busdatas);

%% Measurements
%[V,phi] = newton(ybus,busdatas,linedatas);

%[P,Q,Pij,Qij] = measurements(linedatas,V,phi,ybus);
P = vertcat(P{:});
P = cell2mat(P);
Q = vertcat(Q{:});
Q = cell2mat(Q);
Pij = vertcat(Pij{:});
Pij = cell2mat(Pij);
Qij = vertcat(Qij{:});
Qij = cell2mat(Qij);
P(1:2) = P(1:2)*2;

%% SE initialization

errorlb = -inf*ones(length(P)+length(Q)+length(Pij)+length(Qij),1);
errorub = inf*ones(length(P)+length(Q)+length(Pij)+length(Qij),1);
V_estub = 1.1*ones(length(P),1);
V_estlb = 0.9*ones(length(P),1);
phi_estub = (pi)*ones(length(P),1);
phi_estlb = -(pi)*ones(length(P),1);
phi_estlb(1) = 0;
phi_estub(1) = 0;

x0 = [zeros(length(P)+length(Q)+length(Pij)+length(Qij),1);ones(nbus,1);zeros(nbus,1)];
options = optimset('Display','on','algorithm','sqp');
%tic
[x_old val] = fmincon(@(x) objse(x,P,Q,Pij,Qij),x0,[],[],[],[],[errorlb',V_estlb',phi_estlb'],[errorub',V_estub',phi_estub'],@(x) nlconse(x,P,Q,Pij,Qij,linedatas,ybus),options);
%toc
Vse = x_old(length(P)+length(Q)+length(Pij)+length(Qij)+1:length(P)+length(Q)+length(Pij)+length(Qij)+nbus);
phise = x_old(length(P)+length(Q)+length(Pij)+length(Qij)+nbus+1:length(P)+length(Q)+length(Pij)+length(Qij)+2*nbus);

