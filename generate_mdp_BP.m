function mdp = generate_mdp_BP(beta,omega,alpha,zeta,T)

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
a{1} = A{1};
% Flat distribution for all outcomes when states inconsistent
a{1}(:,1,2) = spm_softmax(zeta*log(A{1}(:,1,2)+exp(-8))); 
a{1}(:,2,3) = spm_softmax(zeta*log(A{1}(:,2,3)+exp(-8)));

%% Transition function: P(S_t| S_t-1, pi)
% =====================================
for f = 1:Nf
    B{f} = zeros(Ns(f));
    b{f} = zeros(Ns(f));
end

% Orientation:
% Generative process:
B{1} = eye(Ns(1)); % Precise transition (i.e., the true state does not change)

% Generative model:
b{1} = spm_norm(eye(Ns(1))+0.5);
b{1} = spm_softmax(omega*b{1}); % b{1} is modulated by omega precision

for f2 = 1:3
            b{2}(f2,:,f2) = 1; % Precise beliefs about actions
end
B{2} = b{2};

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
mdp.beta = beta;
mdp.alpha = alpha;

%labels
label.factor{1} = 'Orientation'; label.name{1} = {'Left', 'Right'};
label.factor{2} = 'Eye position'; label.name{2} = {'Initial','Bottom', 'Top'};
label.modality{1} = 'Features';
label.outcome{1} = {'Null','C1', 'C2'};
label.modality{2} = 'Eye position'; label.outcome{2} = {'Initial', 'Bottom Left', 'Top Right'};
label.action{2} = {'Null','Bottom-Left', 'Top-Right'};
mdp.label = label;
return
