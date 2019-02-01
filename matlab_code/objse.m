function f = objse(h,P,Q,Pij,Qij,nbranch,nbus)

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

f = sum(h(1:length(P_)+length(Q_)+length(Pij_)+length(Qij_)).*h(1:length(P_)+length(Q_)+length(Pij_)+length(Qij_)));