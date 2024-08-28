function AI = calculate_atonia_index(n_MiniEpochs,Fs,toConsider,averageEMGMiniEpochs_corrected)
%Help function to calculate the atonia index

% Create a vector with the miniEpochs to consider
include_miniEpochs = zeros(n_MiniEpochs,1);
for i = 1:n_MiniEpochs
    theseSamples = (i-1)*Fs+1:i*Fs;
    if sum(toConsider(theseSamples))==length(theseSamples) % Only if it is all included
        include_miniEpochs(i) = 1;
    end
end

%EMG values of the miniepochs to consider
averageEMGMiniEpochs_toConsider = averageEMGMiniEpochs_corrected(include_miniEpochs==1);

%Calculate the percentages 
percLess1micro = length(find(averageEMGMiniEpochs_toConsider<=1))/length(averageEMGMiniEpochs_toConsider);
perc1to2micro = length(find(averageEMGMiniEpochs_toConsider>1 & averageEMGMiniEpochs_toConsider<=2))/length(averageEMGMiniEpochs_toConsider);


%Calculate AI according to the definition
AI = percLess1micro/(1-perc1to2micro);
end

