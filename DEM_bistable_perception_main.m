function MDP = DEM_bistable_perception_main

clear; clc

%% Add the SPM file in and define where to store the data
addpath 'D:\PhD\spm12'
addpath 'D:\PhD\spm12\toolbox\DEM'
addpath 'D:\PhD\projects\bistable perception\bistable_perception'
addpath 'D:\PhD\projects\bistable perception\bistable_perception\updates'
path =  'D:\PhD\projects\bistable perception\bistable_perception\results';

if ~exist(path, 'dir')
    mkdir(path)
end



% Set-up:
T    = 32; % Number of time steps
N    = 1; % Number of trials - due to time constraints the reader can decrease it
Nf   = 2;  % Number of factors

% Simulations:
Sim1 = false;  % if true simulate a single trial, false otherwise
% Conditions for Sim1: Only 1 per gamma, omega, and zeta, and N = 1
Sim2 = false; % if true simulate heatmaps and posterior
Figure2 = false; % Generates Figure 2; this has to be switched off for 
% running an entire analysis


% Precision terms:
GAMMA  = [0.001 0.01 0.1 0.2 0.5 1 2 5 10];
OMEGA  = [0.001 0.01 0.1 0.2 0.5 1 2 5 10];
ZETA   = [0.001 0.01 0.1 0.2 0.5 1 2 5 10];
% The reader is encouraged to change these parameters to get more out of
% the model


nze   = numel(ZETA);
nom   = numel(OMEGA);
nga   = numel(GAMMA);

% Initialize matrices
averageswitch = zeros(nom,nga,nze);
rawposterior  = zeros(N, T-1, nom,nga,nze);
post          = zeros(N, T-1, nom,nga,nze);
sc_all        = {};
Nu            = zeros(Nf,1);
% -------------------------------------------------------------------------
% Simulation:
r = 0;
for ze = 1:nze
    for ga = 1:nga
        for om = 1:nom

            rng(r) % simulate different outcomes for every trial
            r     = r + 1;
            omega = OMEGA(om);
            gamma  = GAMMA(ga);
            zeta  = ZETA(ze);

            % Specify model:
            mdp   = generate_mdp_bistable_perception(gamma,omega,zeta,T,Figure2);
            % Specify number of trials:
            M(1:N) = deal(mdp);
            MDP = spm_MDP_VB_X(M);

            % Calculating switch:
            [sc, post, co] = switcher_bistable_perception(MDP,N,T,ze,ga,om,post);
            sc_all.name = sc; % Save when out of the loop
            ave = sum(sc)/N;
            averageswitch(om,ga,ze) = sum(ave);

            if co > 0
                rawposterior = post;
            end

            %% Simuate single trial figure; adapted from spm_MDP_VB_trial
            if Sim1 == true && N == 1 && nga == 1 && nze == 1 && nom == 1
            % To assure only one graph will emerge
                graph = zeros(2,T);
                figure
                for f1 = 1:2
                    for t = 1:T
                        graph(f1,t) = MDP(1).xn{1}(16,f1,t,t);
                    end
                end
                subplot(3,1,1);
                colormap(flipud(gray));
                imagesc(graph);
                hold on
                title('Probability state estimate')
                ylabel('Orientation (s)');
                yticklabels({'','Left', '', 'Right'});
                xlabel('time');
                colorbar;
                caxis([0 1]);

                if iscell(MDP(1).X)
                    Nf = numel(MDP(1).B);  % number of hidden state factors
                    Ng = numel(MDP(1).A);  % number of outcome factors
                    C  = MDP(1).C;
                    for f = 1:Nf
                        Nu(f) = size(MDP(1).B{f},3) > 1;
                    end
                else
                    Nf = 1;
                    Ng = 1;
                    Nu = 1;
                    C  = {MDP(1).C};
                end

                % factors and outcomes to plot
                %----------------------------------------------------------
                maxg  = 3;
                gg = 1:min(Ng,maxg);

                % posterior beliefs about control states
                %----------------------------------------------------------
                Nu     = find(Nu);
                Np     = length(Nu);
                for f  = 1:Np
                    subplot(3,1,2)
                    P = MDP(1).P;
                    if Nf > 1
                        ind     = 1:Nf;
                        for dim = 1:Nf
                            if dim ~= ind(Nu(f))
                                P = sum(P,dim);
                            end
                        end
                        P = squeeze(P);
                    end

                    % display
                    %------------------------------------------------------
                    imagesc(P);
                    colormap(flipud(gray));
                    hold on
                    colorbar;
                    caxis([0 1]);
                    plot(MDP(1).u(Nu(f),:),'.c','MarkerSize',16), hold off
                    title(MDP(1).label.factor{Nu(f)});
                end
                set(gca,'YTick',1:numel(MDP(1).label.action{Nu(f)}));
                title('Posterior policy estimate (with selected eye movements)');
                ylabel('Eye movements (u)');
                set(gca,'YTickLabel',MDP(1).label.action{Nu(f)});
                % sample (observation) and preferences
                %----------------------------------------------------------
                subplot(3,1,3), hold off
                if size(C{gg(2)},1) > 128
                    spm_spy(C{gg(2)},16,1), hold on
                else
                    imagesc(-1*(1 - C{gg(2)})),  hold on
                end
                colormap(flipud(gray));
                colorbar;
                plot(MDP(1).o(gg(1),:),'.c','MarkerSize',16), hold off
                title('Feature outcomes and preferences');
                set(gca,'YTick',1:numel(MDP(1).label.outcome{gg(1)}));
                ylabel('Feature (o)'); xlabel('Time');
                set(gca,'YTickLabel',MDP(1).label.outcome{gg(1)});
            end
        end
    end
end

if Sim2 == true
 bistable_perception_figures4and5(averageswitch,rawposterior,GAMMA,ZETA,OMEGA,N)
end


%% Add in to save
% save(strcat(path,'\averageswitch.mat'),'averageswitch');
% save(strcat(path,'\rawposterior.mat'),'rawposterior');




