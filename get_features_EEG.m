function [f_names,f] = get_features_EEG(EEG,Fs_EEG,stages_30s,window_size)
%get_time_domain_features_EEG: This function returns the EEG features for
%the paper "Machine learning predict phenoconversion from polysomnography
%in isolated REM sleep behavior disorder". The features are calculated in 
%separate windows and their average is obtained across them
%---INPUT:
%- EEG: one EEG signal
%- Fs_EEG: the sampling frequency of the EEG (Hz)
%- stages_30s: the hypnogram. Each sample corresponds to an epoch of 30s. The
%hypnogram is binary, with 1=epoch to consider, 0=epoch not to consider. As
%an example. if you want to calculate the EEG features in REM sleep, all 
%the REM epochs should be scored as 1 and the remaining to 0
%- window_size: the size of the window for feature calculation (s)
%---OUTPUT:
%f_names: the names of the features
%f: the vector containing the average value of the features across the
%windows
addpath("libs\")
%First the signal is selected only in the epochs of interest
stages_30s_rep = repmat(stages_30s,1,30*Fs_EEG);
stages_Fs_EEG = reshape(stages_30s_rep',1,[]);
stages = nan(1,length(EEG));
if length(EEG)>length(stages_Fs_EEG) 
    stages(1:length(stages_Fs_EEG))=stages_Fs_EEG;
else
    stages(1:length(stages)) = stages_Fs_EEG(1:length(stages));
end
this_EEG = EEG(stages==1);
n_windows = floor(length(this_EEG)/(window_size*Fs_EEG));

%Prepare the vectors for calculating features in each window
%Time-domain
z_c = nan(n_windows,1);
hjorth_1 = nan(n_windows,1);
hjorth_2 = nan(n_windows,1);
hjorth_3 = nan(n_windows,1);
tdp_1 = nan(n_windows,1);
tdp_2 = nan(n_windows,1);
perc_diff  = nan(n_windows,1);
coastline = nan(n_windows,1);
root_mean_square = nan(n_windows,1);
variance = nan(n_windows,1);
p2p = nan(n_windows,1);
crest_factor = nan(n_windows,1);
form_factor = nan(n_windows,1);
pulse_indicator = nan(n_windows,1);
mean_rectified = nan(n_windows,1);
%Frequency-domain
peak_power_freq = nan(n_windows,1);
spectral_edge_freq = nan(n_windows,1);
spectral_entropy = nan(n_windows,1);
rel_delta = nan(n_windows,1);
rel_theta = nan(n_windows,1);
rel_alpha = nan(n_windows,1);
rel_beta = nan(n_windows,1);
slow_to_fast_ratio = nan(n_windows,1);
TKE = nan(n_windows,1);
perm_entropy = nan(n_windows,1);
shannon_entropy = nan(n_windows,1);

f_names = {'Zero_crossings','Hjorth_activity','Hjorth_mobility','Hjorth_complexity',...
    'TDP_1','TDP_2','Perc_diff','Coastline','RMS','Var','Peak2Peak','Crest_factor',...
    'Form_factor','Pulse_indicator','Mean_rect',...
    'Peak_power_freq','Spectral_edge_freq','Spectral_entropy',...
    'Rel_delta','Rel_theta','Rel_alpha','Rel_beta','StF_ratio',...
    'TKE','Perm_entropy','Shannon_entropy'};

for i = 1:n_windows
    EEG_window = this_EEG((i-1)*Fs_EEG*window_size+1:i*Fs_EEG*window_size);
    %---------------------------- TIME DOMAIN FEATURES --------------------
    %Zero crossings
    mean_val = nanmean(EEG_window);
    z_c(i) = sum(abs(diff(EEG_window>mean_val)))/length(EEG_window);
    %Hjorth parameters
    [hjorth_1(i),hjorth_2(i),hjorth_3(i)] = hjorth(EEG_window',0);
    %Time-domain properties
    [tdp_1(i),tdp_2(i)] = tdp(EEG_window',0);
    %Percent difference
    perc_diff(i) = prctile(EEG_window,75)-prctile(EEG_window,25);
    %Coastline
    coastline(i) = sum(abs(diff(EEG_window)));
    %Root mean square
    root_mean_square(i) = rms(EEG_window);
    %Variance
    variance(i) = var(EEG_window);
    %Peak to peak
    p2p(i) = peak2peak(EEG_window);
    %Crest factor
    crest_factor(i) = nanmax(abs(EEG_window))/rms(i);
    %Form factor
    form_factor(i) = rms(i)/nanmean(abs(EEG_window));
    %Pulse indicator
    pulse_indicator(i) = nanmax(abs(EEG_window))/nanmean(abs(EEG_window));
    %Mean of rectified value
    mean_rectified(i) = nanmean(abs(EEG_window));
    %-------------------------FREQUENCY DOMAIN FEATURES--------------------
    [Pxx,fxx] = pwelch(EEG_window,Fs_EEG,[],[],Fs_EEG);
    %Peak power frequency
    [~,max_idx] = max(Pxx);
    peak_power_freq(i) = fxx(max_idx);
    %Spectral edge frequency (95%)
    Pdist = cumsum(Pxx);
    power = 0;  
    f_idx = 1;
    while power < Pdist(end)*0.95
        power = sum(Pxx(1:f_idx));
        f_idx=f_idx+1;
    end
    if f_idx > length(fxx)
       f95 = fxx(end);  
    else
        f95 = fxx(f_idx);    
    end
    spectral_edge_freq(i) = f95;
    %Spectral entropy
    NormPsd = Pxx/Pdist(end);
    spectral_entropy(i) = -sum(NormPsd.*log(NormPsd));    
    if isnan(spectral_entropy(i))
        spectral_entropy(i) = 0;
    end
    %Relative power in different bands
    total = [0.5 35];
    f_total = fxx>=total(1) & fxx<total(2);
    total_power = trapz(fxx(f_total),Pxx(f_total));
    delta = [0.5 4]; theta = [4 8]; alpha = [8 13]; beta = [13 35];
    f_delta = fxx>=delta(1) & fxx<delta(2);
    f_theta = fxx>=theta(1) & fxx<theta(2);
    f_alpha = fxx>=alpha(1) & fxx<alpha(2);
    f_beta = fxx>=beta(1) & fxx<beta(2);
    rel_delta(i) = trapz(fxx(f_delta),Pxx(f_delta))/total_power;
    rel_theta(i) = trapz(fxx(f_theta),Pxx(f_theta))/total_power;
    rel_alpha(i) = trapz(fxx(f_alpha),Pxx(f_alpha))/total_power;
    rel_beta(i) = trapz(fxx(f_beta),Pxx(f_beta))/total_power;
    slow_to_fast_ratio(i) = (rel_delta(i)+rel_theta(i))/(rel_alpha(i)+rel_beta(i));
    %-------------------------NON-LINEAR FEATURES--------------------
    TKE(i) = mean(TKEO(EEG_window));
    perm_entropy(i) = petropy(EEG_window,10,1,'order');
    shannon_entropy(i) = Entropy_Array(EEG_window);
end

f(1) = nanmean(z_c);
f(2) = nanmean(hjorth_1);
f(3) = nanmean(hjorth_2);
f(4) = nanmean(hjorth_3);
f(5) = nanmean(tdp_1);
f(6) = nanmean(tdp_2);
f(7) = nanmean(perc_diff);
f(8) = nanmean(coastline);
f(9) = nanmean(root_mean_square);
f(10) = nanmean(variance);
f(11) = nanmean(p2p);
f(12) = nanmean(crest_factor);
f(13) = nanmean(form_factor);
f(14) = nanmean(pulse_indicator);
f(15) = nanmean(mean_rectified);
f(16) = nanmean(peak_power_freq);
f(17) = nanmean(spectral_edge_freq);
f(18) = nanmean(spectral_entropy);
f(19) = nanmean(rel_delta);
f(20) = nanmean(rel_theta);
f(21) = nanmean(rel_alpha);
f(22) = nanmean(rel_beta);
f(23) = nanmean(slow_to_fast_ratio);
f(24) = nanmean(TKE);
f(25) = nanmean(perm_entropy);
f(26) = nanmean(shannon_entropy);

end