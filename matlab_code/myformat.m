function [busdatas linedatas gencost] = myformat(testsystem)
data = testsystem;
%% Busdata
busdatas = data.bus(:,1:2);
busdatas(find(busdatas(:,2)==3),2)=0;
busdatas(find(busdatas(:,2)==1),2)=3;
busdatas(find(busdatas(:,2)==0),2)=1;
busdatas(:,3)=data.bus(:,8);
busdatas(:,4:12)=zeros(length(data.bus(:,1)),9);
busdatas(data.gen(:,1),5:6)=[data.gen(:,2),data.gen(:,3)];
busdatas(:,7:8)=data.bus(:,3:4);
busdatas(data.gen(:,1),9:10)=[data.gen(:,5),data.gen(:,4)];
busdatas(:,11:12)=[data.bus(:,5),data.bus(:,6)];
%% Linedata
linedatas = [data.branch(:,1:5),data.branch(:,9)];
linedatas(find(linedatas(:,6)==0),6) = 1;
%%Gencost
gencost = [data.gen(:,1),data.gencost(:,5:7),data.gen(:,10),data.gen(:,9)];
