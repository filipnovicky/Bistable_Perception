function MDP = DEM_demo_MDP_BP_degeneracy

clear; clc
rng('default') 
addpath D:\PhD\Code\spm
addpath D:\PhD\Supervision\Bistable_Perception\Code
addpath   D:\PhD\Code\spm\toolbox\DEM
path =  'D:\PhD\Supervision\Bistable_Perception\Code';

if ~exist(path, 'dir')
    mkdir(path)
end
%{
   A theory for the model:
   - A switch during bistable perception is enactivist in its
   core and modulary under selection of different precision terms
   triggering explorative actions over the stimulus (e.g., Necker's cube)

   A model for bistable perception:
   -  3 One-step policies (Fixate_1; Fixate_2)
   -  Two hidden state factors: 2xOrientation + 3xFixation, and
   -  Two outcome modalities: 3xFeatures + 3xEye Position
   -  Modulation of precision for A Matrix (zeta), B matrix (omega), and
   policy selection (beta)
   -  Switches are counted when MDP.xn crosses the probability of 0.5 for
   an orientation between the current and previous time step and is
   accompanied with a cognruent action

   The goal of the simulation:
   -  Explain the relationship between different precisions on perceiving
   bistable perception
   - Provide a proof of the concept for i) an enactivist perspective on
   bistable perception; ii) relate attentional mechanisms and neural
   network to bistable perception; iii) explain the role of neuromodulatory
   circuitry to bistable perception

%}

% Set-up:
T = 32; % Number of time steps
N = 1; % number of trials
Nf=2;
Sim1 = 0; % if 1: simulate a single trial
Sim2 = 0; % if 1: generate the 'a' matrices (N == 1)
Sim3 = 0; % if 1: simulate graphs of averages
Sim4 = 0; % if 1: generate the 'b' matrices (N == 1)
Sim5 = 1; % if 1: generate average posterior probability for switches per zetas
Sim6 = 0; % if 1: generate a plot with average switches per zeta

BETA = 0.1; % 0.1:0.75:3.1;
OMEGA = 6;% 0:2:8;
ZETA = [0:0.1:0.5 1 2 3 4 5];
alpha = 64;
nze = numel(ZETA);
nom = numel(OMEGA);
nbe = numel(BETA);

% Initialize future matrices
averageswitch = zeros(nom,nbe,nze);
avredundancy = zeros(nom,nbe,nze);
aventropy= zeros(nom,nbe,nze);
avenergy = zeros(nom,nbe,nze);
avcost = zeros(nom,nbe,nze);
stdemat = zeros(nom,nbe,nze);
avepost = zeros(nze,1);
comat = zeros(nze,1);
sc_all = {};
ave_sc = {};

% Simulation:
r = 0;
for ze = 1:nze
    for be = 1:nbe
        for om = 1:nom
            rng(r) % simulate different outcomes for every trial
            r = r + 1;
            omega = OMEGA(om);
            beta = BETA(be);
            zeta = ZETA(ze);
            % Specify model name:
            % name = strcat(num2str(be),'_',num2str(om),'_',num2str(ze),'_',num2str(alpha));
            
            % Specify model:
            mdp = generate_mdp_17_05(beta,omega,alpha,zeta,T);
            % Specify number of trials:
            M(1:N) = deal(mdp);
            
            % Simulate:
            MDP = spm_MDP_VB_X(M);
            % Save the model - not necessary now
            % save(strcat(path,'\',name,'.mat'),'MDP');
            
            % Trial level summands: 
            for XN = 1:N
                [MDPentropy{XN},  MDPenergy{XN}, MDPcost{XN}, MDPaccuracy{XN}, MDPredundancy{XN}] = free_energy_decomp(MDP(XN));
            end
           % Averaged summands: 

            aventropy(om,be,ze) = average_quantity(MDPentropy, Nf, T, N,1);
            avenergy(om,be,ze) = average_quantity(MDPenergy,Nf, T, N,1);
            avcost(om,be,ze) = average_quantity(MDPcost, Nf, T, N,1);
            avredundancy(om,be,ze) = average_quantity(MDPredundancy, Nf, T, N,1);
            
            % Calculating switch:
            [sc, post, co] = switcher2505(MDP,N,T,ze,nze);
            sc_all.name = sc; % Save when out of the loop
            ave = sum(sc)/N;
            ave_sc.name = sum(sc)/N;
            stde = std(sum(sc,2));
            averageswitch(om,be,ze) = sum(ave);
            stdemat(om,be,ze) = stde;
            % average of posterior beliefs
            if co > 0
                avepost(ze,1) = sum(post(:,ze))/co;
            end
            comat(ze) = co/N;
            %% Simuate single trial figure
            if Sim1 == 1
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
                ylabel('orientation'); % 1 = Left; 2 = Right
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
                if nargin < 2, gf = 1:min(Nf,maxg); end
                if nargin < 3, gg = 1:min(Ng,maxg); end
                nf    = numel(gf);
                ng    = numel(gg);
                
                % posterior beliefs about control states
                %----------------------------------------------------------
                Nu     = find(Nu);
                Np     = length(Nu);
                title({'Omega ', omega, 'Zeta ', zeta, 'Beta ', beta});
                
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
                set(gca,'XTickLabel',{});
                set(gca,'XTick',1:size(X{1},2));
                set(gca,'YTick',1:numel(MDP(1).label.action{Nu(f)}));
                set(gca,'YTickLabel',MDP(1).label.action{Nu(f)});
                % sample (observation) and preferences
                %----------------------------------------------------------
                subplot(3,1,3), hold off
                if size(C{gg(1)},1) > 128
                    spm_spy(C{gg(1)},16,1), hold on
                else
                    imagesc(1 - C{gg(1)}),  hold on
                end
                plot(MDP(1).o(gg(1),:),'.c','MarkerSize',16), hold off
                colorbar;
                title(sprintf('Outcomes and preferences - %s',MDP(1).label.modality{gg(1)}));
                set(gca,'XTickLabel',{});
                set(gca,'XTick',1:size(X{1},2))
                set(gca,'YTick',1:numel(MDP(1).label.outcome{gg(1)}));
                set(gca,'YTickLabel',MDP(1).label.outcome{gg(1)});
            end
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
                    % subplot(1,2,i-1);
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
                figure
                % how to run the figures only once when there are more
                % precision terms?
                title({'Transition function for Omega: ', omega});
                colormap(flipud(gray));
                hold on
                ylabel('Orientation at tau');
                xlabel('Orientation at tau+1');
                xticks([1 2])
                xticklabels({'Left', 'Right'})
                colorbar;
                yticks([1 2])
                yticklabels({'Left', 'Right'})
                caxis([0 1]);
                y = MDP.b{1};
                imagesc(y)
            end
        end
    end
    %% Simulate average-switch graphs (the code adapted from MATLAB's Q&A)
    if Sim3 == 1
        figure
        bar(averageswitch(:,:,ze), 'grouped');
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
        xlim([0 6]);
        xticklabels(0:2:8);
        xlabel('Omega');
        ylim([0 22]);
        ylabel('Number of Switches')
        legend('beta = 0.1','beta = 0.85','beta = 1.6','beta = 2.35', 'beta = 3.1')
    end
end
if Sim5 == 1
    figure
    plot(avepost) % simulate posteriors for different zetas
    title('Average Posterior probability of switches');
    ylim([0 1]);
    xlim([ZETA(1)+1 nze]);
    xticklabels(ZETA);
    xlabel('Zeta');
    ylabel('Posterior switch probability')
end
if Sim6 == 1
    figure
    plot(comat, 'r')
    title('Average number of switches');
    ylim([0 20]);
    xlim([ZETA(1)+1 nze]);
    xticklabels(ZETA);
    xlabel('Zeta');
    ylabel('Number of switches')
% save(strcat(path,'\sc_all.mat'),'sc_all');
% save(strcat(path,'\ave_sc.mat'),'ave_sc');
% save(strcat(path,'\averageswitch.mat'),'averageswitch');
% save(strcat(path,'\stdemat.mat'),'stdemat');
end