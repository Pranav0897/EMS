function [Vse phise]=state_estimate(path_to_file, P,Q,Pij,Qij, V_up, V_d)
%% MATPOWER to MYFORMAT
% [busdatas, linedatas, gencost] = myformat(ext2int(casefile));
casefile=load(path_to_file);
casefile=casefile.a_dict;
[busdatas, linedatas, gencost] = myformat(casefile);
nbus = max(max(linedatas(:,1)),max(linedatas(:,2)));
nbranch=length(linedatas(:,1));


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

%% Filtering missing data

P_ = [];
Q_ = [];
Pij_ = [];
Qij_ = [];

for i=1:nbus
    if(isnan(P(i)))
        continue;
    end
    P_ = [P_, P(i)];
    Q_ = [Q_, Q(i)];
end

for i=1:nbranch
    if(isnan(Pij(i)))
        continue;
    end
    Pij_ = [Pij_, Pij(i)];
    Qij_ = [Qij_, Qij(i)];
end
%% SE initialization

errorlb = -inf*ones(length(P_)+length(Q_)+length(Pij_)+length(Qij_),1);
errorub = inf*ones(length(P_)+length(Q_)+length(Pij_)+length(Qij_),1);
V_estub = V_up*ones(nbus,1);
V_estlb = V_d*ones(nbus,1);
phi_estub = (pi)*ones(nbus,1);
phi_estlb = -(pi)*ones(nbus,1);
phi_estlb(1) = 0;
phi_estub(1) = 0;

x0 = [zeros(length(P_)+length(Q_)+length(Pij_)+length(Qij_),1);ones(nbus,1);zeros(nbus,1)];
options = optimset('Display','on','algorithm','sqp');
%tic
[x_old val] = fmincon(@(x) objse(x,P,Q,Pij,Qij,nbranch,nbus),x0,[],[],[],[],[errorlb',V_estlb',phi_estlb'],[errorub',V_estub',phi_estub'],@(x) nlconse(x,P,Q,Pij,Qij,linedatas,ybus),options);
%toc
Vse = x_old(length(P_)+length(Q_)+length(Pij_)+length(Qij_)+1:length(P_)+length(Q_)+length(Pij_)+length(Qij_)+nbus);
phise = x_old(length(P_)+length(Q_)+length(Pij_)+length(Qij_)+nbus+1:length(P_)+length(Q_)+length(Pij_)+length(Qij_)+2*nbus);

