%Sample file to test state_estimate function. Not needed when creating the python package, so you can exclude this then

close all
clear all
clc
%% MATPOWER to MYFORMAT
%case14 busdata, linedata, gencost
case14 = load('case14.mat');
case14 = case14.a_dict;
[busdatas, linedatas, gencost] = myformat(case14);
nbus = max(max(linedatas(:,1)),max(linedatas(:,2)));
%% Ybus calculation
[ybus A] = ybus_incidence(linedatas,busdatas);
%% Measurements
[V,phi] = newton(ybus,busdatas,linedatas);
[P,Q,Pij,Qij] = measurements(linedatas,V,phi,ybus);
% save('P')
% save('Q')
% save('Pij')
% save('Qij')
[Vse,phise]= state_estimate(P,Q,Pij,Qij);
ybus;
A;


