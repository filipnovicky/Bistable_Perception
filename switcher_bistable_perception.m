function [sc, post, co] = switcher_bistable_perception(MDP,N,T,ze,ga,om,post)

%{
   Three conditions to counting switches: 
         1) higher probability in the left orientation hidden state in the
         current time step
         2) high probability in the right orientation hidden state in the
         previous time step
%}


th = 0.5+1e-32; % Probability threshold for counting a switch
co = 0; % counting the switches



if N==1
    s = zeros(1,2);
    for f1 = 1:2
        for t = 2:T
            y = MDP.xn{1}(16,f1,t,t);
            z = MDP.xn{1}(16,f1,t-1,t-1);

            if f1 == 1 && y > th && z < 1 - th
                sc1 = 1;
                sc2 = 0;
                post(1,t-1,om,ga,ze) = post(1,t-1,om,ga,ze) + y; % save posterior beliefs
                co = co + 1;

            elseif f1 == 2 && y > th && z < 1 - th
                sc2 = 1;
                sc1 = 0;
                post(1,t-1,om,ga,ze) = post(1,t-1,om,ga,ze) + y;
                co = co + 1;
            else
                sc1 = 0;
                sc2 = 0;
                post(1,t-1,om,ga,ze) = post(1,t-1,om,ga,ze) + 0;
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
                y = MDP(n).xn{1}(16,f1,t,t);
                z = MDP(n).xn{1}(16,f1,t-1,t-1);

                if f1 == 1 && y > th && z < 1 - th
                    sc1 = 1;
                    sc2 = 0;
                    post(n,t-1,om,ga,ze) = post(n,t-1,om,ga,ze) + y;
                    co = co + 1;

                elseif f1 == 2 && y > th && z < 1 - th
                    sc2 = 1;
                    sc1 = 0;
                    post(n,t-1,om,ga,ze) = post(n,t-1,om,ga,ze) + y;
                    co = co + 1;

                else
                    sc1 = 0;
                    sc2 = 0;
                    post(n,t-1,om,ga,ze) = post(n,t-1,om,ga,ze) + 0;
                end
                s(1,1) = s(1,1) + sc1;
                s(1,2) = s(1,2) + sc2;

            end
        end
        sc(n,:) = s;
    end
end

return