function [sc, post, co] = switcher2505(MDP,N,T,ze,nze)

%{
   Three conditions to counting switches: 
         1) higher probability in the left orientation hidden state in the
         current time step
         2) high probability in the right orientation hidden state in the
         previous time step
         3) Changes in orientation are intertwined with their related
         fixation points (actions) (e.g., left orientation = top-right
         fixation)

Output is sc  [zeros(N,2)]: the switchcount matrix(number of trials, for different switches)

%}

th = 0.5; % 0.5+1e-16; % Probability threshold for counting a switch
post = zeros(T-1,nze);
co = 0;
if N==1
    s = zeros(1,2);
    for f1 = 1:2
        for t = 2:T
            y = MDP.xn{1}(16,f1,t,t);
            z = MDP.xn{1}(16,f1,t-1,t-1);
            u = MDP.u(2,t-1);
            if f1 == 1 && y > th && z < 1 - th && u == 2
                sc1 = 1;
                sc2 = 0;
                post(t-1,ze) = post(t-1,ze) + y; % save posterior beliefs
                co = co + 1;
            elseif f1 == 2 && y > th && z < 1 - th && u == 1
                sc2 = 1;
                sc1 = 0;
                post(t-1,ze) = post(t-1,ze) + y;
                co = co + 1;
            else
                sc1 = 0;
                sc2 = 0;
                post(t-1,ze) = post(t-1,ze) + 0;
            end
            s(1,1) = s(1,1) + sc1; 
            s(1,2) = s(1,2) + sc2; 
        end
    end
    sc = s;

else
    
    sc = zeros(N,2); 
    for n = 1:N
        %% switchcount
        s = zeros(1,2);
        for f1 = 1:2 % 1:Ns(1)
            for t = 2:T
                % We are looking at the orientations - i.e., 1
                y = MDP(n).xn{1}(16,f1,t,t);
                z = MDP(n).xn{1}(16,f1,t-1,t-1);
                u = MDP(n).u(2,t-1);
                if f1 == 1 && y > th && z < 1 - th && u == 2
                    sc1 = 1;
                    sc2 = 0;
                    post(t-1,ze) = post(t-1,ze) + y;
                    co = co + 1;
                elseif f1 == 2 && y > th && z < 1 - th && u == 1
                    sc2 = 1;
                    sc1 = 0;
                    post(t-1,ze) = post(t-1,ze) + y;
                    co = co + 1;
                else
                    sc1 = 0;
                    sc2 = 0;
                    post(t-1,ze) = post(t-1,ze) + 0;
                end
                s(1,1) = s(1,1) + sc1;
                s(1,2) = s(1,2) + sc2;
            end
        end
        sc(n,:) = s;
    end
end