function atonia_index = get_atonia_index(EMG,Fs,stages)
%GET_ATONIA_INDEX This function calculates the atonia index for the
%sleep stages identified in the vector stages.
%---INPUT
%- EMG: the EMG signal (whole night)
%- Fs: the sampling frequency of the EMG signal (Hz)
%- stages: the hypnogram. Same size as EMG. The
%hypnogram is binary, with 1=epoch to consider, 0=epoch not to consider. As
%an example. if you want to calculate the REM atonia index, all 
%the samples corresponding to REM sleep should be scored as 1, while the
%rest 0
%---OUTPUT
%- atonia index: the atonia index for the samples defined in "stages"

%1. Rectify the signal
EMG_rect = 2*abs(EMG);

%2. Now divide the whole signals into 1s mini-epochs
n_MiniEpochs = floor(length(EMG_rect)/Fs);

%3. Calculate the average in each mini-epoch
averageEMGMiniEpochs = nan(n_MiniEpochs,1);
for j = 1:n_MiniEpochs
    theseSamples = (j-1)*Fs+1:j*Fs;
    this_EMG = EMG_rect(theseSamples);
    averageEMGMiniEpochs(j) = nanmean(this_EMG);
end

%4. Make the noise correction
averageEMGMiniEpochs_corrected = nan(n_MiniEpochs,1);
%Noise correction for the first 30 miniEpochs
for j = 1:30
    thisWindow = 1:j+30;
    minimumThisWindow = nanmin(averageEMGMiniEpochs(thisWindow));
    averageEMGMiniEpochs_corrected(j) = averageEMGMiniEpochs(j)-minimumThisWindow;
end
%Noise correction fot the middle miniEpochs
for j = 31:length(averageEMGMiniEpochs_corrected)-30
    thisWindow = j-30:j+30;
    minimumThisWindow = nanmin(averageEMGMiniEpochs(thisWindow));
    averageEMGMiniEpochs_corrected(j) = averageEMGMiniEpochs(j)-minimumThisWindow;
end
%Noise correction for the last 30 miniEpochs
for j = length(averageEMGMiniEpochs_corrected)-29:length(averageEMGMiniEpochs_corrected)
    thisWindow = j-30:length(averageEMGMiniEpochs_corrected);
    minimumThisWindow = nanmin(averageEMGMiniEpochs(thisWindow));
    averageEMGMiniEpochs_corrected(j) = averageEMGMiniEpochs(j)-minimumThisWindow;
end

%5.Calculate the indices in the different stages
atonia_index = calculate_atonia_index(n_MiniEpochs,Fs,stages,averageEMGMiniEpochs_corrected);

end

