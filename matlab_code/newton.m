function [V,phi] = newton(Ybus_nr,busdatas,linedatas)

linedatas = vertcat(linedatas{:});
linedatas = cell2mat(linedatas);
Ybus_nr = vertcat(Ybus_nr{:});
Ybus_nr = cell2mat(Ybus_nr);
busdatas = vertcat(busdatas{:});
busdatas = cell2mat(busdatas);

%% Newton-Raphson Load Flow 
%% Getting busdata
nbus = length(busdatas(:,1));
baseMVA = 100;

type = busdatas(:,2); % Type of Bus 1-Slack, 2-PV, 3-PQ
V = busdatas(:,3); % Slach Voltage and Voltage mag intitials
Vsp = busdatas(:,3); % Slach Voltage and Voltage mag intitials
phi = busdatas(:,4); % Voltage Angle intitials
Pg = busdatas(:,5)/baseMVA;
Qg = busdatas(:,6)/baseMVA;
Pl = busdatas(:,7)/baseMVA;
Ql = busdatas(:,8)/baseMVA;
Qmin = busdatas(:,9)/baseMVA; % Minimum Reactive Power Limit..
Qmax = busdatas(:,10)/baseMVA; % Maximum Reactive Power Limit..
Psp = Pg - Pl; % calculate powers in the busses: P Specified
Qsp = Qg - Ql; % calculate powers in the busses: Q Specified
pq = find(type == 3); % PQ Buses(there is no generation)
pv = find(type == 2 | type == 1); % PV Buses
npq = length(pq); % No. of PQ buses
G_nr = real(Ybus_nr);
B_nr = imag(Ybus_nr);
Tol = 1;
itr = 1;
%% Iteration Starts:
while (Tol > 1e-9 && itr < 100)
    P = zeros(nbus,1);
    Q = zeros(nbus,1);
    % Calculate P and Q
    for i = 1:nbus
        for k = 1:nbus
            P(i) = P(i) + V(i)* V(k)*(G_nr(i,k)*cos(phi(i)-phi(k)) + B_nr(i,k)*sin(phi(i)-phi(k))); % pp. 77 Wang. (eq 2.9)
            Q(i) = Q(i) + V(i)* V(k)*(G_nr(i,k)*sin(phi(i)-phi(k)) - B_nr(i,k)*cos(phi(i)-phi(k))); % ...
        end
    end
    % Checking Q-limit violations..
    if itr <= 7 && itr > 2 % Only checked up to 7th iterations..
        for n = 2:nbus
            if type(n) == 2
                QG = Q(n)+Ql(n);
                if QG < Qmin(n)
                    V(n) = V(n) + 0.01;
                elseif QG > Qmax(n)
                    V(n) = V(n) - 0.01;
                end
            end
        end
    end
    dP = Psp-P; % Calculate change from specified value pp.78 Wang. (eq2.13)
    dQ1 = Qsp-Q; % ...
    k = 1;
    dQ = zeros(npq,1);
    for i = 1:nbus
        if type(i) == 3
            dQ(k,1) = dQ1(i);
            k = k+1;
        end
    end
    r = [dP(2:nbus); dQ]; % Mismatch Vector, not considering the first value that is the slack bus P,Q
    %% The Jacobian matrix
    % J1 - Derivative of Real Power Injections with Angles
    J1 = zeros(nbus-1,nbus-1);
    for i = 1:(nbus-1)
        m = i+1;
        for k = 1:(nbus-1)
            n = k+1;
            if n == m
                for n = 1:nbus
                    J1(i,k) = J1(i,k) - V(m)* V(n)*(G_nr(m,n)*sin(phi(m)-phi(n)) - B_nr(m,n)*cos(phi(m)-phi(n))); % pp. 84 Wang. eq(2.41)
                end
                J1(i,k) = J1(i,k) - V(m)^2*B_nr(m,m);
            else
                J1(i,k) = V(m)* V(n)*(G_nr(m,n)*sin(phi(m)-phi(n)) - B_nr(m,n)*cos(phi(m)-phi(n))); % pp. 84 Wang. eq(2.42)
            end
        end
    end
    % J2 - Derivative of Real Power Injections with V
    J2 = zeros(nbus-1,npq);
    for i = 1:(nbus-1)
        m = i+1;
        for k = 1:npq
            n = pq(k);
            if n == m
                for n = 1:nbus
                    J2(i,k) = J2(i,k) + V(n)*(G_nr(m,n)*cos(phi(m)- phi(n)) + B_nr(m,n)*sin(phi(m)-phi(n)));
                end
                J2(i,k) = J2(i,k) + V(m)*G_nr(m,m);
            else
                J2(i,k) = V(m)*(G_nr(m,n)*cos(phi(m)-phi(n)) + B_nr(m,n)*sin(phi(m)-phi(n)));
            end
        end
    end
    % J3 - Derivative of Reactive Power Injections with Angles
    J3 = zeros(npq,nbus-1);
    for i = 1:npq
        m = pq(i);
        for k = 1:(nbus-1)
            n = k+1;
            if n == m
                for n = 1:nbus
                    J3(i,k) = J3(i,k) + V(m)*V(n)*(G_nr(m,n)*cos(phi(m)-phi(n)) + B_nr(m,n)*sin(phi(m)-phi(n)));
                end
                J3(i,k) = J3(i,k) - V(m)^2*G_nr(m,m);
            else
                J3(i,k) = V(m)* V(n)*(-G_nr(m,n)*cos(phi(m)-phi(n)) - B_nr(m,n)*sin(phi(m)-phi(n))); % pp. 84 Wang. eq(2.44) !!!!! MANFI
            end
        end
    end
    % J4 - Derivative of Reactive Power Injections with V
    J4 = zeros(npq,npq);
    for i = 1:npq
        m = pq(i);
        for k = 1:npq
            n = pq(k);
            if n == m
                for n = 1:nbus
                    J4(i,k) = J4(i,k) + V(n)*(G_nr(m,n)*sin(phi(m)- phi(n)) - B_nr(m,n)*cos(phi(m)-phi(n)));
                end
                J4(i,k) = J4(i,k) - V(m)*B_nr(m,m);
            else
                J4(i,k) = V(m)*(G_nr(m,n)*sin(phi(m)-phi(n)) - B_nr(m,n)*cos(phi(m)-phi(n))); % pp. 85 Wang. eq(2.48) !!!!! MANFI
            end
        end
    end
    J = [J1 J2;
        J3 J4]; % Jacobian Matrix % J = Jacob(busdatas,ybus,nbus);
    X = J\r; % inv(J)*r; <TIME SAVING> % Correction Vector
    dTh = X(1:nbus-1); % Change in Voltage Angle
    dV = X(nbus:end); % Change in Voltage Magnitude
    %% record the phita V and phita Angle
    dV_sq(itr,:) = dV';
    dTh_sq(itr,:) = dTh';
    %% Updating State Vectors
    phi(2:nbus) = dTh + phi(2:nbus); % Angle update
    k = 1;
    for i = 1:nbus
        if type(i) == 3
            V(i) = dV(k) + V(i); % Voltage Magnitude update
            k = k+1;
        else
            V(i) = Vsp(i); % reset the slack and PV bus voltages to the specified values
        end
    end
    itr = itr + 1; % iteration counter
    Tol = max(abs(r));
end % end of Iterations
% fprintf('N-R Iterations = %4d', itr);fprintf('\n');
%% Figure of convergence
% figure
% plot([1:itr-1],diag(dV_sq*dV_sq'),[1:itr-1],diag(dTh_sq*dTh_sq'),'g');
% title('Load Flow: phita-V and phita-Phi decrease according toIterations'); xlabel('iteration'); ylabel('phita V & Angle'); grid on %figure for convergence
end