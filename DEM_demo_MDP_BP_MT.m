function MDP = DEM_demo_MDP_BP_MT

clear; clc
addpath '\\unimaas.nl\users\Students\i6256962\data\My Documents\MATLAB\spm12'
addpath '\\unimaas.nl\users\Students\i6256962\data\My Documents\model'
addpath '\\unimaas.nl\users\Students\i6256962\data\My Documents\MATLAB\spm12\toolbox\DEM'
path =  '\\unimaas.nl\users\Students\i6256962\data\My Documents\results';

if ~exist(path, 'dir')
    mkdir(path)
end
%{
Model theory:
   - A switch during bistable perception is enactivist in its
   core and modulatory under selection of different precision terms
   triggering explorative actions over the stimulus (e.g., Necker's cube)
Bistable perception model:
   -  3 One-step policies (Fixate at Top-Right; Fixate at Bottom-Left; Fixate at Null)
   -  Two hidden state factors: 2xOrientation + 3xFixation, and
   -  Two outcome modalities: 3xFeatures + 3xEye Position
   -  Modulation of precision of A Matrix (zeta), B matrix (omega), and
   expected free energy (beta)
   -  Switches are counted when MDP.xn crosses the probability of 0.5 for
   an orientation between the current and previous time step and is
   accompanied with a congruent action
Hypothesis and direction:
   -  Explain the relationship between different precisions on perceiving
   bistable perception and that this phenomenon is brought about by
   multiple possibilities regulated via the model's redundancy and entropy
   - Provide a proof of the concept for
        i) an enactivist perspective on bistable perception
            - a different perception when a different action are selected
        ii) the relation of attentional mechanisms and neural network to
            bistable perception
            - FBA controls the release of precision-modulating
            neurotransmitters
%}

% Set-up:
T    = 32; % Number of time steps
N    = 32;  % Number of trials - due to time constraints the reader can decrease it
Nf   = 2;  % Number of factors
% Simulations:
Sim1 = 0;  % if 1: simulate a single trial (N == 1)
Sim2 = 0;  % if 1: generate the geneartive model's likelihood matrix (N == 1)
Sim3 = 0;  % if 1: simulate graphs of averages
Sim4 = 0;  % if 1: generate the geneartive model's transition matrix (N == 1)
Sim5 = 0;  % if 1: generate average posterior probability for switches per zeta
Sim6 = 0;  % if 1: generate a plot with average switches per zeta
Sim7 = 0;  % if 1: simulate degeneracy
Sim8 = 0;  % if 1: simulate redundancy
Sim9 = 0;  % if 1: simulate free energy

% Precision terms (can be changed to explore and understand the model better):
BETA  = [0.1 2 7];
OMEGA = [0.1 2 7];
ZETA  = [0.1 0.3 0.5];
alpha = 64; % Precision over actions determining stochastic/deterministic behaviour

nze   = numel(ZETA);
nom   = numel(OMEGA);
nbe   = numel(BETA);

% Initialize matrices
averageswitch = zeros(nom,nbe,nze);
avredundancy  = zeros(nom,nbe,nze);
aventropy     = zeros(nom,nbe,nze);
stdentropy    = zeros(nom,nbe,nze);
stdredundancy = zeros(nom,nbe,nze);
avaccuracy    = zeros(nom,nbe,nze);
stdaccuracy   = zeros(nom,nbe,nze);
avcost        = zeros(nom,nbe,nze);
stdcost       = zeros(nom,nbe,nze);
avfe          = zeros(nom,nbe,nze);
stdfe         = zeros(nom,nbe,nze);
stdemat       = zeros(nom,nbe,nze);
avepost       = zeros(nze,1);
comat         = zeros(nze,1);
postmat       = zeros(nom,nbe,nze);
sc_all        = {};
ave_sc        = {};
Nu            = zeros(Nf,1);
% -------------------------------------------------------------------------
% Simulation:
r = 0;
for ze = 1:nze
    for be = 1:nbe
        for om = 1:nom
            rng(r) % simulate different outcomes for every trial
            r     = r + 1;
            omega = OMEGA(om);
            beta  = BETA(be);
            zeta  = ZETA(ze);

            % Specify model:
            mdp   = generate_mdp_BP(beta,omega,alpha,zeta,T);
            % Specify number of trials:
            M(1:N) = deal(mdp);
            MDP = spm_MDP_VB_X(M);
            % Simulate Entropy and Redundancy:
            for XN = 1:N
                [MDPentropy{XN},  MDPenergy{XN}, MDPcost{XN}, MDPaccuracy{XN}, MDPredundancy{XN}] = free_energy_decomp(MDP(XN));
            end

            % Averaged summands:
            [entrav,entrst] = average_quantity(MDPentropy, Nf, T, N);
            aventropy(om,be,ze) = entrav;
            stdentropy(om,be,ze) = entrst;
            [costav, coststd] = average_quantity(MDPcost, Nf, T, N);
            avcost(om,be,ze) = costav;
            stdcost(om,be,ze) = coststd;
            [accav,accstd] = average_quantity(MDPaccuracy, Nf, T, N);
            avaccuracy(om,be,ze) = accav;
            stdaccuracy(om,be,ze) = accstd;
            [redav,redstd] = average_quantity(MDPredundancy, Nf, T, N);
            avredundancy(om,be,ze) = redav;
            stdredundancy(om,be,ze) = redstd;
            avfe(om,be,ze) = redav - accav;
            stdfe(om,be,ze) = accstd + redstd;
            
            % Calculating switch:
            [sc, post, co] = switcher_BP(MDP,N,T,ze,nze);
            sc_all.name = sc; % Save when out of the loop
            ave = sum(sc)/N;
            ave_sc.name = sum(sc)/N;
            stde = std(sum(sc,2));
            averageswitch(om,be,ze) = sum(ave);
            postmat(ze,1) = mean(post(:,ze));
            stdemat(om,be,ze) = stde;
            % average probability of posterior beliefs
            if co > 0
                avepost(ze,1) = sum(post(:,ze))/co;
            end
            comat(ze,1) = co/N;
            
            %% Simuate single trial figure; adapted from spm_MDP_VB_trial
            if Sim1 == 1 && N == 1
                graph = zeros(2,T);
                figure
                for f1 = 1:2 % 1:Ns(1)
                    for t = 1:T
                        graph(f1,t) = MDP(1).xn{1}(16,f1,t,t);
                    end
                end
                subplot(3,1,1);
                colormap(flipud(gray));
                imagesc(graph);
                hold on
                title('Probability distribution over orientation')
                ylabel('orientation');
                ylim([1 2]);
                yticklabels({'Left', 'Right'});
                xlabel('time');
                colorbar;
                caxis([0 1]);
                
                if iscell(MDP(1).X)
                    Nf = numel(MDP(1).B);  % number of hidden state factors
                    Ng = numel(MDP(1).A);  % number of outcome factors
                    X  = MDP(1).X;
                    C  = MDP(1).C;
                    for f = 1:Nf
                        Nu(f) = size(MDP(1).B{f},3) > 1;
                    end
                else
                    Nf = 1;
                    Ng = 1;
                    Nu = 1;
                    X  = {MDP(1).X};
                    C  = {MDP(1).C};
                end
                
                % factors and outcomes to plot
                %----------------------------------------------------------
                maxg  = 3;
                if nargin < 3
                    gg = 1:min(Ng,maxg);
                end
                
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
                title(sprintf('Outcomes and preferences - %s',MDP(1).label.modality{gg(1)}));
                set(gca,'YTick',1:numel(MDP(1).label.outcome{gg(1)}));
                set(gca,'YTickLabel',MDP(1).label.outcome{gg(1)});
            end
            %--------------------------------------------------------------
            if Sim2 == 1 && N == 1
                for i = 2:3
                    figure
                    if i == 1 % Initial fixation is always flat
                        title({'Zeta ', zeta,'Initial fixation'});
                    elseif i == 2
                        title({'Zeta ', zeta,'Bottom-Left'});
                    elseif i == 3
                        title({'Zeta ', zeta,'Top-Right'});
                    end
                    colormap(flipud(gray));
                    hold on
                    ylabel('Features');
                    xlabel('Orientation');
                    xticks([1 2])
                    xticklabels({'Left', 'Right'})
                    colorbar;
                    yticks([1 2 3])
                    yticklabels({'C2','C1','Initial'})
                    caxis([0 1]);
                    x = MDP.a{1}(:,:,i);
                    imagesc(flipud(x))
                end
            end
            if Sim4 == 1 && N == 1
                figure('Name','Transition function','NumberTitle','off')
                title({'Omega', omega});
                colormap(flipud(gray));
                hold on
                ylabel('Current time-step');
                xlabel('Next time-step');
                xticks([1 2])
                xticklabels({'Left', 'Right'})
                colorbar;
                yticks([1 2])
                yticklabels({'Left', 'Right'})
                caxis([0 1]);
                y = MDP.b{1};
                imagesc(flipud(y))
            end
            % -------------------------------------------------------------
        end
    end
    %% Simulate average-switch graphs (the code adapted from MATLAB's Q&A)
    if Sim3 == 1
        figure
        bar(averageswitch(:,:,ze), 'grouped');
        grid
        hold on
        % Find the number of groups and the number of bars in each group
        [ngroups, nbars] = size(averageswitch(:,:,ze));
        % Calculate the width for each bar group
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        % Set the position of each error bar in the centre of the main bar
        % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
        for i = 1:nbars
            % Calculate center of each bar
            x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
            errorbar(x,averageswitch(:,i,ze),stdemat(:,i,ze),'k','LineStyle','none');
        end
        hold off
        title('Zeta ', zeta);
        xlim([0 4]);
        xticklabels({'0.1', '2', '7'});
        xlabel('Omega');
        ylim([0 22]);
        ylabel('Number of Switches')
        legend('beta = 0.1','beta = 2', 'beta = 7')
    end
    
    if Sim7 == 1 % Entropy simulation (same algorithm as above for comments)
        figure
        bar(aventropy(:,:,ze), 'grouped');
        grid
        hold on
        [ngroups, nbars] = size(aventropy(:,:,ze));
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        for i = 1:nbars
            x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
            errorbar(x,aventropy(:,i,ze),stdentropy(:,i,ze),'k','LineStyle','none');
        end
        hold off
        title('Zeta ', zeta);
        xlim([0 4]);
        xticklabels({'0.1', '2', '7'});
        xlabel('Omega');
        ylim([0 30]);
        ylabel('Average Degeneracy (nats)')
        legend('beta = 0.1','beta = 2', 'beta = 7')
    end
    if Sim8 == 1 % Redundancy simulation (same algorithm as above for comments)
        figure
        bar(avredundancy(:,:,ze), 'grouped');
        grid
        hold on
        [ngroups, nbars] = size(avredundancy(:,:,ze));
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        for i = 1:nbars
            x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
            errorbar(x,avredundancy(:,i,ze),stdredundancy(:,i,ze),'k','LineStyle','none');
        end
        hold off
        title('Zeta ', zeta);
        xlim([0 4]);
        xticklabels({'0.1', '2', '7'});
        xlabel('Omega');
        ylim([0 5]);
        ylabel('Average Redundancy (nats)')
        legend('beta = 0.1','beta = 2', 'beta = 7')
    end
    if Sim9 == 1 % generate free energy (same algorithm as above for comments)
        figure
        bar(avfe(:,:,ze), 'grouped');
        grid
        hold on
        [ngroups, nbars] = size(avfe(:,:,ze));
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        for i = 1:nbars
            x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
            errorbar(x,avfe(:,i,ze),stdfe(:,i,ze),'k','LineStyle','none');
        end
        hold off
        title('Zeta ', zeta);
        xlim([0 4]);
        xticklabels({'0.1' '2' '7'});
        xlabel('Omega');
        ylim([0 20]);
        ylabel('Average Free Energy (nats)')
        legend('beta = 0.1','beta = 2', 'beta = 7')
    end
    % ---------------------------------------------------------------------
end
if Sim5 == 1
    figure
    plot(avepost)
    grid
    title('Average Posterior probability of switches');
    ylim([0 1]);
    xlim([ZETA(1)+1 nze]);
    xticks([1 2 3])
    xticklabels(ZETA);
    xlabel('Zeta');
    ylabel('Posterior switch probability');
end
if Sim6 == 1
    figure
    plot(comat)
    grid
    title('Average number of switches');
    ylim([0 22]);
    xlim([ZETA(1) + 1 nze]);
    xticks([1 2 3])
    xticklabels(ZETA);
    xlabel('Zeta');
    ylabel('Number of switches');
end
save(strcat(path,'\sc_all.mat'),'sc_all');
save(strcat(path,'\ave_sc.mat'),'ave_sc');
save(strcat(path,'\averageswitch.mat'),'averageswitch');
save(strcat(path,'\stdemat.mat'),'stdemat');