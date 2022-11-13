
%% Heatmaps:
function bistable_perception_figures4and5(averageswitch,rawposterior,GAMMA,ZETA,OMEGA,N)


%% Initialize data
try
    GAMMA;
    ZETA;
    OMEGA;
    N;
catch
    GAMMA  = [0.001 0.01 0.1 0.2 0.5 1 2 5 10];
    OMEGA  = [0.001 0.01 0.1 0.2 0.5 1 2 5 10];
    ZETA   = [0.001 0.01 0.1 0.2 0.5 1 2 5 10];
    N = 64;
end


%% Figure 4
figure
try
    data = averageswitch;
catch
    data = load('D:\PhD\projects\bistable perception\bistable_perception\results\averageswitch.mat', 'averageswitch');
    % Add the location of the data file
    data = data.averageswitch;
end

num = size(data(:,:,1),2);
for i = 1:num

    [X,Y] = meshgrid(1:size(data(i,:,:),2), 1:size(data(i,:,:),3));

    %// Define a finer grid of points
    [X2,Y2] = meshgrid(1:0.01:size(data(i,:,:),2), 1:0.01:size(data(i,:,:),3));

    %// Interpolate the data and show the output
    data2 = reshape(data(i,:,:), [numel(GAMMA),numel(ZETA)]);
    outData = interp2(X, Y, data2, X2, Y2, 'linear');

    subplot(3,3,i);

    imagesc(outData);

    title(strcat('\omega = ', num2str(OMEGA(i))),'fontsize', 14);
    set(gca, 'XTick', linspace(1,size(X2,2),size(X,2)));
    set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
    set(gca, 'XTickLabel', ZETA);
    xlabel('\zeta', 'fontsize', 20);
    ylabel('\gamma', 'fontsize', 20);
    set(gca, 'YTickLabel', GAMMA);

    colorbar;
    caxis([0 15])
end

%% Figure 5:


for l = 1:2 %% A and B figures
    zetapost = zeros(numel(ZETA),1);
    zetastd = zetapost;
    gammapost = zeros(numel(GAMMA),1);
    gammastd = gammapost;
    omegapost = zeros(numel(OMEGA),1);
    omegastd = omegapost;
    figure
    if l == 1
        try
            data = rawposterior;
        catch
            data = load('D:\PhD\projects\bistable perception\bistable_perception\results\rawposterior.mat', 'rawposterior');
            data = data.rawposterior;
        end
        ylabel('Average posterior switch probability','fontsize', 18,'FontName', 'Times')

    elseif l == 2
        try
            data = averageswitch;
        catch
            data = load('D:\PhD\projects\bistable perception\bistable_perception\results\averageswitch.mat', 'averageswitch');
            data = data.averageswitch;
        end
        ylabel('Average number of switches','fontsize', 18,'FontName', 'Times')
    end
    
    % Find the average values
    for i = 1:numel(ZETA)
        for j = 1:numel(GAMMA)
            for k = 1:numel(OMEGA)
                if l == 1
                    zetapost(i,1)  = mean(nonzeros(data(:,:,:,:,i)));
                    gammapost(j,1) = mean(nonzeros(data(:,:,:,j,:)));
                    omegapost(k,1) = mean(nonzeros(data(:,:,k,:,:)));

                elseif l == 2
                    zetapost(i,1)  = mean2(data(:,:,i));
                    gammapost(j,1) = mean2(data(:,j,:));
                    omegapost(k,1) = mean2(data(k,:,:));

                end

            end
        end
    end

    x = (1:numel(ZETA))'; % The values on the x-axis -- works for gamma and omega as well

    % Plot the averages and define the fitted regression models
    for a = 1:3
        if a == 1
            y = zetapost;
            if l == 1
                [f,gof_zetapost] = fit(x,y,'poly2');

                h = plot(f,'b',x,y,'bdiamond');
            elseif l == 2
                [f,gof_zetaswitch] = fit(x,y,'poly3');

                h = plot(f,'b',x,y,'bdiamond');
            end
            set(h,'LineWidth',3.5)
            hold on

        elseif a == 2
            y = omegapost;
            if l == 1
                [f,gof_omegapost] = fit(x,y,'poly3');
                h = plot(f,'g',x,y,'gsquare');
            elseif l == 2
                [f,gof_omegaswitch] = fit(x,y,'poly2');

                h = plot(f,'g',x,y,'gsquare');
            end
            set(h,'LineWidth',3.5)
            hold on
        elseif a == 3

            y = gammapost;
            [f,gof_gammapost] = fit(x,y,'poly1');
            h = plot(f,'c',x,y,'c^');
            set(h,'LineWidth',3.5)

        end
        
        box on
        set(gca, 'XTick', linspace(1,9,9),'FontName','Times')
        xticklabels(OMEGA)

        if l == 1
            ylabel('Posterior switch Probability','fontsize', 18,'FontName','Times')
            xlabel('Precision values','fontsize', 18,'FontName','Times')
            ylim([0.5 0.85])
            legend('$\zeta$','$2^{nd} order p.$','$\omega$', '$3^{rd} order p.$', ...
            '$\gamma$','$1^{st} order p.$','Interpreter', 'latex','Location','northwest','fontsize', 11,'FontName','Times')

        elseif l == 2
            ylabel('Number of switches','fontsize', 18,'FontName','Times')
            xlabel('Precision values','fontsize', 18,'FontName','Times')
            legend('$\zeta$','$3^{rd} order p.$','$\omega$', '$2^{nd} order p.$', ...
            '$\gamma$','$1^{st} order p.$','Interpreter', 'latex','Location','northeast','fontsize', 11,'FontName','Times')
            ylim([0 20])

        end
    end
end