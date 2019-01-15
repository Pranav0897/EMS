%% Formulation of Ybus by singular transformation method (With Transformer Tap settings and Shunt Admittances)
function [Ybus, A] = ybus_incidence(linedatas,busdatas)
%global fb tb nbranch nbus linedatas baseMVA busdatas
r = linedatas(:,3);
x = linedatas(:,4);
b = linedatas(:,5);
tap = linedatas(:,6); % Tap setting values (one for the other buses)
fb = linedatas(:,1);
tb = linedatas(:,2);
nbus = length(busdatas(:,1));
nbranch = length(linedatas(:,1));
baseMVA =100;
GS = busdatas(:,11); % shunt conductance (MW at V = 1.0 p.u.)
BS = busdatas(:,12); % shunt susceptance (MW at V = 1.0 p.u.)
Ysh = ( GS + 1j * BS) / baseMVA; % vector of shunt admittances
Z= r + 1i*x; % z matrix...
Y = 1./Z;
%% Formation of Bus Incidence matrix A (signs: comes in is -1, goes out is +1)
A=zeros(nbranch+nbus,nbus);
for i=1:nbus % building top I submatrix:
    for j=1:nbus
        if(i==j)
            A(i,i)=1;
        end
    end
end
for i = nbus+1 : nbus+nbranch % building Buttom A_branch submatrix:
    A( i , fb(i-nbus)) = 1;
    A( i , tb(i-nbus)) = -1;
end
%% Calculation of primitive matrix
Yprimitive = zeros(nbranch+nbus,1);
% For buses:
for i=1:nbranch
    Yprimitive(fb(i)) = Yprimitive(fb(i)) + 1i*b(i)/2 + (1-tap(i)) * Y(i) / tap(i)^2;
    Yprimitive(tb(i)) = Yprimitive(tb(i)) + 1i*b(i)/2 + (tap(i)-1) * Y(i) / tap(i);
end
Yprimitive(1:nbus) = Yprimitive(1:nbus) + Ysh; % adding shunt admittances
% Branches:
for i=1:nbranch
    Yprimitive(i+nbus) = Y(i) / tap(i);
end
%% Bus Admittance matrix:
Ybus = A' * diag(Yprimitive) * A; %% shunt admittance
