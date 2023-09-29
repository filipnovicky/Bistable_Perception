function mdp = generate_mdp_bistable_perception(gamma,omega,zeta,T,Figure2)

%Generate Figure 2 - it has to be switched off for the entire analysis
% Figure2 = False;

if Figure2 == true
    gamma = 1;
    omega = 0.1;
    zeta = 0.1;
    T = 32;
end

%% Prior beliefs: P(s_o)
% =============================
D{1} = [0.5 0.5]';  % Orientation {Left, Right}
D{2} = [1 0 0]';  % Fixation {Null, Bottom-Left, Top-Right}

% Number of Factors:
Nf = numel(D);

% Number of levels for each factor:
for f = 1:Nf
    Ns(f) = numel(D{f});
end

% Outcome modalities:
% ----------------------------------
% Features {Null, C3, C6}
% Eye Position {Null, Bottom Left, Top Right}

No    = [3 3];
Ng    = numel(No);
%% Likelihood function: P(o_t | s_t):
% ======================================
% Initialising likelihood:
for g = 1:Ng
    A{g} = zeros([No(g),Ns]);
    a{g} = zeros([No(g),Ns]);
end

for f1 = 1:Ns(1) % S1 Hidden State Orientation {Left, Right}
    for f2 = 1:Ns(2) % S2 Hidden State Location {Null, Bottom-Left, Top-Right}
        for f = 1:Nf
            A{f}(f2,:,f2) = 1; % All outcomes are independent of orientation
            a{2}(f2,:,f2) = 1; % actions are precise for the generative model
        end
    end
end
a{1} = spm_softmax(0*A{1})*100;

% Flat distribution for all outcomes when states are inconsistent
a{1}(:,1,3) = spm_softmax(zeta*log(A{1}(:,1,3)+exp(-8)))*100;
a{1}(:,2,2) = spm_softmax(zeta*log(A{1}(:,2,2)+exp(-8)))*100;

% The multiplication with 100 assures stronger concentration parameters
% used as precision parameters



%% Transition function: P(S_t| S_t-1, pi)
% =====================================
for f = 1:Nf
    B{f} = zeros(Ns(f));
    b{f} = zeros(Ns(f));
end

% Orientation:
% Generative process:
B{1} = eye(Ns(1));

% Generative model:
b{1} = spm_softmax(omega*log(B{1}+exp(-8)))*100;



if Figure2 == true
    n = 1; nn  =6; nnn=11;
    colormap(flipud(gray));
    for i = [0.001,0.01, 0.1,0.2,0.5]
        a{1} = spm_softmax(a{1});
        a{1}(:,2,2) = spm_softmax(i*log(A{1}(:,2,2)+exp(-8)));
        a{1}(:,1,3) = spm_softmax(i*log(A{1}(:,1,3)+exp(-8)));

        subplot(3,5,n);
        imagesc(a{1}(:,:,2))
        caxis([0 1])
        xticks([1 2])
        xticklabels({'Left', 'Right'})
        title(strcat('\zeta = ', num2str(i)),'fontsize', 14);
        yticks([1 2 3]);
        if n == 1
            yticklabels({'Null', 'Corner 1', 'Corner 2'});
            ylabel('Feature (o)');
        else
            yticklabels({'', '', ''});
        end
        n = n + 1;
        subplot(3,5,nn);
        imagesc(a{1}(:,:,3))
        caxis([0 1])
        title(strcat('\zeta = ', num2str(i)),'fontsize', 14);
        xticks([1 2])
        xticklabels({'Left', 'Right'})
        yticks([1 2 3]);
        if nn==6
            yticklabels({'Null', 'Corner 1', 'Corner 2'})
            ylabel('Feature (o)');
        else
            yticklabels({'', '', ''});
        end
        if nn == 6; xlabel('Orientation (s)'); end
        nn= nn +1;

        subplot(3,5,nnn)
        y =spm_softmax( i*log(B{1}+exp(-8)));
        colormap(flipud(gray));
        box on
        hold on
        imagesc(flipud(y))
        title(strcat('\omega = ', num2str(i)),'fontsize', 14);
        xticks([1.25 1.75])
        xticklabels({'Left', 'Right'})
        xlim([1 2])
        yticks([1 2]);
        caxis([0 1]);
        if nnn==11
            yticklabels({'Right', 'Left'})
            ylabel('Orientation (s_{t+1})');
        else
            yticklabels({'', '', ''});
        end
        if nnn == 11; xlabel('Orientation (s_{t})'); end
        nnn= nnn+1;
        hold on
    end
end




for f2 = 1:3
    b{2}(f2,:,f2) = 100; % Precise beliefs about actions
end
B{2} = b{2}*100;

% C matrix so there is no action towards null fixation
C{1} = zeros(3,T);
for t = 1:T
    C{2}(:,t) = [-20
        0
        0];
end

% Specify 1-step Policies [u_1, u_2, u_3]:
%--------------------
U(:,:,1) = [1 1 1];
U(:,:,2) = [1 2 3];


%% MDP structure
%=================================================
mdp.T = T;
mdp.U = U;
mdp.A = A;
mdp.a = a;
mdp.B = B;
mdp.b = b;
mdp.C = C;
mdp.D = D;
mdp.eta = 0;
mdp.beta = 1/gamma;

%labels
label.factor{1} = 'Orientation'; label.name{1} = {'Left', 'Right'};
label.factor{2} = 'Eye position'; label.name{2} = {'Initial','Bottom', 'Top'};
label.modality{1} = 'Features';
label.outcome{1} = {'Null','C1', 'C2'};
label.modality{2} = 'Eye position'; label.outcome{2} = {'Initial', 'Bottom Left', 'Top Right'};
label.action{2} = {'Initial','Bottom-Left', 'Top-Right'};
mdp.label = label;
return
