function [c ceq]=nlconse(h,P,Q,Pij,Qij,linedatas,ybus)
nbus = max(max(linedatas(:,1)),max(linedatas(:,2)));
nbranch=length(linedatas(:,1));

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

nbus_ = length(P_);

Vse = h(length(P_)+length(Q_)+length(Pij_)+length(Qij_)+1:length(P_)+length(Q_)+length(Pij_)+length(Qij_)+nbus);
phise = h(length(P_)+length(Q_)+length(Pij_)+length(Qij_)+nbus+1:length(P_)+length(Q_)+length(Pij_)+length(Qij_)+2*nbus);
b = linedatas(:,5);
fb = linedatas(:,1);
tb = linedatas(:,2);
G = real(ybus);
B = imag(ybus);
Pcal = zeros(nbus_,1);
Qcal = zeros(nbus_,1);


dl = 1;

for l = 1:nbus
    if isnan(P(l))
        continue;
    end
    for m = 1:nbus
        Pcal(dl,1) = Pcal(dl) + Vse(l)* Vse(m)*(G(l,m)*cos(phise(l)-phise(m)) + B(l,m)*sin(phise(l)-phise(m)));
        Qcal(dl,1) = Qcal(dl) + Vse(l)* Vse(m)*(G(l,m)*sin(phise(l)-phise(m)) - B(l,m)*cos(phise(l)-phise(m))); 
    end
    dl=dl+1;
end

bbus_perturbed = zeros(nbus,nbus);
for k=1:nbranch
    bbus_perturbed(fb(k),tb(k)) = b(k) ./ 2;
    bbus_perturbed(tb(k),fb(k)) = bbus_perturbed(fb(k),tb(k));
end

del_o = 1;
%% power flows calculation
for  o = 1:nbranch
    if(isnan(Pij(o)))
        continue;
    end

    m = fb(o);
    n = tb(o);
    Pijcal(del_o,1) = -Vse(m)^2*(G(m,n)) + Vse(m)*Vse(n)*(G(m,n)*cos(phise(m)-phise(n))+ B(m,n)*sin(phise(m)-phise(n)));
    Qijcal(del_o,1) = Vse(m)^2*(B(m,n)- bbus_perturbed(m,n)) +Vse(m)*Vse(n)*(G(m,n)*sin(phise(m)-phise(n)) - B(m,n)*cos(phise(m)-phise(n)));
    del_o=del_o+1;
end

    P_ = P_';
    Q_ = Q_';
    Pij_ = Pij_';
    Qij_ = Qij_';

    ceq = [Pcal-P_-h(1:length(P_));Qcal-Q_-h(length(P_)+1:length(P_)+length(Q_));Pijcal-Pij_-h(length(P_)+length(Q_)+1:length(P_)+length(Q_)+length(Pij_));Qijcal-Qij_-h(length(P_)+length(Q_)+length(Pij_)+1:length(P_)+length(Q_)+length(Pij_)+length(Qij_))];
    c =[];