function f = objse(h,P,Q,Pij,Qij)
f = sum(h(1:length(P)+length(Q)+length(Pij)+length(Qij)).*h(1:length(P)+length(Q)+length(Pij)+length(Qij)));