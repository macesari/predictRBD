function [f_names,f] = get_features_EMG(EMG,Fs_EMG,stages_30s,window_size)
%get_time_domain_features_EMG: This function returns the EMG features for
%the paper "Machine learning predict phenoconversion from polysomnography
%in isolated REM sleep behavior disorder". The features are calculated in 
%separate windows and their average is obtained across them
%---INPUT:
%- EMG: one EMG signal
%- Fs_EMG: the sampling frequency of the EMG (Hz)
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
stages_30s_rep = repmat(stages_30s,1,30*Fs_EMG);
stages_Fs_EMG = reshape(stages_30s_rep',1,[]);
stages = nan(1,length(EMG));
if length(EMG)>length(stages_Fs_EMG) 
    stages(1:length(stages_Fs_EMG))=stages_Fs_EMG;
else
    stages(1:length(stages)) = stages_Fs_EMG(1:length(stages));
end
this_EMG = EMG(stages==1);
n_windows = floor(length(this_EMG)/(window_size*Fs_EMG));


%Define the feature names and prepare the vectors for the features
f_names = {'Atonia_index','Energy','Perc_75_abs','Shannon_entropy',...
    'Shannon_entropy_abs','Fractal_exponent','Abs_gamma_power','Spectral_edge_frequency',...
    'Peak_power_frequency','Spectral_entropy',...
    'Zero_crossings','RMS','Var','Peat_to_peak','Crest_factor',...
    'Form_factor','Pulse_indicator','TKE','Perm_entropy','Wilson_amplitude',...
    'Myopulse_indicator','Integral','Wavelength'};
energy = nan(n_windows,1);
perc_75 = nan(n_windows,1);
shannon_entropy = nan(n_windows,1);
shannon_entropy_abs = nan(n_windows,1);
fractal_exponent = nan(n_windows,1);
abs_gamma_power = nan(n_windows,1);
spectral_edge_freq = nan(n_windows,1);
peak_power_freq = nan(n_windows,1);
spectral_entropy = nan(n_windows,1);
z_c = nan(n_windows,1);
root_mean_square = nan(n_windows,1);
variance = nan(n_windows,1);
p2p = nan(n_windows,1);
crest_factor = nan(n_windows,1);
form_factor = nan(n_windows,1);
pulse_indicator = nan(n_windows,1);
TKE = nan(n_windows,1);
perm_entropy = nan(n_windows,1);
wilson_amplitude = nan(n_windows,1);
myopulse = nan(n_windows,1);
integral = nan(n_windows,1);
wavelength = nan(n_windows,1);

%Threshold for calculating some of the features
th = nanmean(this_EMG) + 3*nanstd(this_EMG);

for i = 1:n_windows
    EMG_window = this_EMG((i-1)*Fs_EMG*window_size+1:i*Fs_EMG*window_size);
    %Energy
    energy(i) = sum(EMG_window.^2);
    %75th percentile of rectified signal
    perc_75(i) = prctile(abs(EMG_window),75);
    %Shannon entropy
    shannon_entropy(i) = Entropy_Array(EMG_window);
    %Shannon entropy rectified
    shannon_entropy_abs(i) = Entropy_Array(abs(EMG_window));
    %Fractal exponent
    NFFT = 2^nextpow2(length(EMG_window));
    [pxx, fxx] = periodogram(EMG_window,hann(length(EMG_window)),NFFT,Fs_EMG);
    fxx_rng_idx = fxx<=100 & fxx>=10;
    fxx_rng = fxx(fxx_rng_idx);
    pxx_rng = pxx(fxx_rng_idx);   
    log_pow = log(pxx_rng);
    log_freq = log(fxx_rng);
    fractal_exponent(i) = -1*(log_freq\log_pow);
    %EMG absolute gamma power
    abs_gamma_power(i) = bandpower(pxx,fxx,[30 45],'psd');
    %Spectral edge frequency (95%)
    Pdist = cumsum(pxx);
    power = 0;  
    f_idx = 1;
    while power < Pdist(end)*0.95
        power = sum(pxx(1:f_idx));
        f_idx=f_idx+1;
    end
    if f_idx > length(fxx)
       f95 = fxx(end);  
    else
        f95 = fxx(f_idx);    
    end
    spectral_edge_freq(i) = f95;
    %Peak power frequency
    [~,max_idx] = max(pxx);
    peak_power_freq(i) = fxx(max_idx);
    %Spectral entropy
    NormPsd = pxx/Pdist(end);
    spectral_entropy(i) = -sum(NormPsd.*log(NormPsd));    
    if isnan(spectral_entropy(i))
        spectral_entropy(i) = 0;
    end
    %Zero crossings
    mean_val = nanmean(EMG_window);
    z_c(i) = sum(abs(diff(EMG_window>mean_val)))/length(EMG_window);
    %Root mean square
    root_mean_square(i) = rms(EMG_window);
    %Variance
    variance(i) = var(EMG_window);
    %Peak to peak
    p2p(i) = peak2peak(EMG_window);
    %Crest factor
    crest_factor(i) = nanmax(abs(EMG_window))/rms(i);
    %Form factor
    form_factor(i) = rms(i)/nanmean(abs(EMG_window));
    %Pulse indicator
    pulse_indicator(i) = nanmax(abs(EMG_window))/nanmean(abs(EMG_window));
    %TKE
    TKE(i) = mean(TKEO(EMG_window));
    %Permutation entropy
    perm_entropy(i) = petropy(EMG_window,10,1,'order');
    %Wilson amplitude
    x = abs(diff(EMG_window));
    wilson_amplitude(i) = sum(x>=th)/length(EMG_window);
    %Myopulse indicator
    myopulse(i) = sum(EMG_window>=th)/length(EMG_window);
    %Integral
    integral(i) = sum(abs(EMG_window))/length(EMG_window);
    %Wavelenght
    wavelength(i) = sum(abs(diff(EMG_window)))/length(EMG_window);
end

f(1) = get_atonia_index(EMG,Fs_EMG,stages);
f(2) = nanmean(energy);
f(3) = nanmean(perc_75);
f(4) = nanmean(shannon_entropy);
f(5) = nanmean(shannon_entropy_abs);
f(6) = nanmean(fractal_exponent);
f(7) = nanmean(abs_gamma_power);
f(8) = nanmean(spectral_edge_freq);
f(9) = nanmean(peak_power_freq);
f(10) = nanmean(spectral_entropy);
f(11) = nanmean(z_c);
f(12) = nanmean(root_mean_square);
f(13) = nanmean(variance);
f(14) = nanmean(p2p);
f(15) = nanmean(crest_factor);
f(16) = nanmean(form_factor);
f(17) = nanmean(pulse_indicator);
f(18) = nanmean(TKE);
f(19) = nanmean(perm_entropy);
f(20) = nanmean(wilson_amplitude);
f(21) = nanmean(myopulse);
f(22) = nanmean(integral);
f(23) = nanmean(wavelength);


end