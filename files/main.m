%%% PART 1 %%%
% Step 1: Load the audio file
[audio, Fs] = audioread('voice.wav');

% Step 2: Plot the audio signal
figure;
if size(audio, 2) == 2
    % Stereo audio
    subplot(2, 1, 1);
    plot(audio(:, 1));
    title('Left Channel');
    xlabel('Sample Index');
    ylabel('Amplitude');

    subplot(2, 1, 2);
    plot(audio(:, 2));
    title('Right Channel');
    xlabel('Sample Index');
    ylabel('Amplitude');
else
    % Mono audio
    plot(audio);
    title('Mono Channel');
    xlabel('Sample Index');
    ylabel('Amplitude');
end

% Step 3: Double the sampling rate
Fs_2 = Fs * 2;

% Step 4: Save the audio with doubled sampling rate
audiowrite('out_doubled.wav', audio, Fs_2);

% Explanation
disp('The audio data has been saved with a doubled sampling rate. This makes the audio play faster and at a higher pitch.');


%%% PART 2 %%%

% Load the audio file
[audio, Fs] = audioread('voice.wav');

% Create a decreasing exponential signal
n = length(audio);
t = (0:n-1)' / Fs;
exp_signal = exp(-t);

% If the audio is stereo, ensure the exponential signal matches both channels
if size(audio, 2) == 2
    exp_signal = [exp_signal, exp_signal];
end

% Multiply the audio signal by the exponential signal
modified_audio = audio .* exp_signal;

% Plot the modified audio signal
figure;
if size(modified_audio, 2) == 2
    % Stereo audio
    subplot(2, 1, 1);
    plot(modified_audio(:, 1));
    title('Modified Left Channel');
    xlabel('Sample Index');
    ylabel('Amplitude');

    subplot(2, 1, 2);
    plot(modified_audio(:, 2));
    title('Modified Right Channel');
    xlabel('Sample Index');
    ylabel('Amplitude');
else
    % Mono audio
    plot(modified_audio);
    title('Modified Mono Channel');
    xlabel('Sample Index');
    ylabel('Amplitude');
end

% Save the modified audio signal
audiowrite('modified_audio.wav', modified_audio, Fs);

% Explanation
disp('The modified audio data has been saved with the volume decreasing exponentially over time.');


%%% PART 3 %%%

% Step 1: Load the audio file
[audio, Fs] = audioread('voice.wav');

% Define echo parameters
first_echo_delay = 1.0; % seconds
second_echo_delay = 2.0; % seconds
first_echo_intensity = 0.8;
second_echo_intensity = 0.5;

% Step 2: Create the impulse response
n = length(audio);
first_echo_samples = round(first_echo_delay * Fs);
second_echo_samples = round(second_echo_delay * Fs);

% Impulse response length
h_length = max([n, first_echo_samples + n, second_echo_samples + n]);
impulse_response = zeros(h_length, 1);
impulse_response(1) = 1; % Original signal
impulse_response(first_echo_samples + 1) = first_echo_intensity; % First echo
impulse_response(second_echo_samples + 1) = second_echo_intensity; % Second echo

% Step 3: Convolve the audio with the impulse response
% If stereo, process each channel separately
if size(audio, 2) == 2
    modified_audio_left = conv(audio(:, 1), impulse_response);
    modified_audio_right = conv(audio(:, 2), impulse_response);
    modified_audio = [modified_audio_left(1:n), modified_audio_right(1:n)];
else
    modified_audio = conv(audio, impulse_response);
    modified_audio = modified_audio(1:n); % Ensure the length matches the original audio
end

% Normalize the modified audio to avoid clipping
modified_audio = modified_audio / max(abs(modified_audio));

% Step 4: Save the modified audio
audiowrite('modified_audio_with_echo.wav', modified_audio, Fs);

% Explanation
disp('The audio data has been convolved with the echo filter and saved to "modified_audio_with_echo.wav".');


%%% PART 4 %%%

% Step 1: Load the impulse response for the concert hall
[ir_concert_hall, Fs_ir_concert_hall] = audioread('concert_hall_IR.wav');

% Ensure the impulse response is a column vector
ir_concert_hall = ir_concert_hall(:);

% Plot the impulse response for the concert hall
figure;
plot(ir_concert_hall);
title('Impulse Response - Concert Hall');
xlabel('Sample Index');
ylabel('Amplitude');

% Assuming audio and Fs are still in the workspace from previous steps
% Convolve the audio with the concert hall impulse response
if size(audio, 2) == 2
    % Stereo audio
    modified_audio_concert_hall_left = conv(audio(:, 1), ir_concert_hall, 'same');
    modified_audio_concert_hall_right = conv(audio(:, 2), ir_concert_hall, 'same');
    modified_audio_concert_hall = [modified_audio_concert_hall_left, modified_audio_concert_hall_right];
else
    % Mono audio
    modified_audio_concert_hall = conv(audio, ir_concert_hall, 'same');
end

% Normalize the modified audio to avoid clipping
modified_audio_concert_hall = modified_audio_concert_hall / max(abs(modified_audio_concert_hall), [], 'all');

% Save the modified audio for the concert hall impulse response
audiowrite('modified_audio_concert_hall.wav', modified_audio_concert_hall, Fs);

% Step 2: Load the impulse response for the iron bucket
[ir_iron_bucket, Fs_ir_iron_bucket] = audioread('iron_bucket_IR.wav');

% Ensure the impulse response is a column vector
ir_iron_bucket = ir_iron_bucket(:);

% Plot the impulse response for the iron bucket
figure;
plot(ir_iron_bucket);
title('Impulse Response - Iron Bucket');
xlabel('Sample Index');
ylabel('Amplitude');

% Convolve the audio with the iron bucket impulse response
if size(audio, 2) == 2
    % Stereo audio
    modified_audio_iron_bucket_left = conv(audio(:, 1), ir_iron_bucket, 'same');
    modified_audio_iron_bucket_right = conv(audio(:, 2), ir_iron_bucket, 'same');
    modified_audio_iron_bucket = [modified_audio_iron_bucket_left, modified_audio_iron_bucket_right];
else
    % Mono audio
    modified_audio_iron_bucket = conv(audio, ir_iron_bucket, 'same');
end

% Normalize the modified audio to avoid clipping
modified_audio_iron_bucket = modified_audio_iron_bucket / max(abs(modified_audio_iron_bucket), [], 'all');

% Save the modified audio for the iron bucket impulse response
audiowrite('modified_audio_iron_bucket.wav', modified_audio_iron_bucket, Fs);

disp('The audio has been convolved with both impulse responses and saved to respective files.');


%%% PART 5 %%%
%% Function Definitions
function y = trapezoidal_rule(func, ll, ul, n)
    dx = (ul - ll) / n;
    x = linspace(ll, ul, n + 1);
    y = sum(func(x) .* ([1, 2 * ones(1, n - 1), 1] * dx / 2));
end


%% Task 5: Fourier series
% Define the original function x(t) as a repetitive step function
T = 4;  % Period of the function
x = @(t) mod(t, T) < T / 4 | mod(t, T) >= 3 * T / 4;

% Define the shifted function x_shifted(t)
x_shifted = @(t) x(t - T/4);

% Define the interval for integration
ll = -T / 2;
ul = T / 2;

% Define the range of harmonics
k_ranges = {-20:20, -100:100, -500:500};

% Initialize arrays to store results
fourier_series_results = cell(size(k_ranges));
t = linspace(-2*T, 2*T, 10000);  % Evaluate over a larger range for better visualization

% Function to calculate Fourier coefficients using trapezoidal rule
calc_fourier_coeffs = @(k_range) arrayfun(@(k) trapezoidal_rule(@(t) x(t) .* exp(-1i * k * 2 * pi * t / T), ll, ul, 100000) / T, k_range);

for idx = 1:length(k_ranges)
    k_range = k_ranges{idx};
    coeffs = calc_fourier_coeffs(k_range);
    fourier_series = zeros(size(t));

    for k_idx = 1:length(k_range)
        k = k_range(k_idx);
        fourier_series = fourier_series + coeffs(k_idx) * exp(1i * k * 2 * pi * t / T - 1i * k * pi / 2);
    end

    fourier_series_results{idx} = real(fourier_series);  % Store the real part
end

% Plot results for different harmonic ranges
figure;
for idx = 1:length(k_ranges)
    subplot(length(k_ranges), 1, idx);
    plot(t, fourier_series_results{idx});
    hold on;
    plot(t, x_shifted(t), 'r');
    title(sprintf('Fourier Series with %d Harmonics', max(abs(k_ranges{idx}))));
    legend('Fourier Series', 'Shifted Function');
    hold off;
end

% Define the new function x_new(t)
x_new = @(t) mod(t, T) < T / 4 | (mod(t, T) >= T / 2 & mod(t, T) < 3 * T / 4);

% Calculate the new Fourier series coefficients
k_range = -500:500;
coeffs_new = arrayfun(@(k) trapezoidal_rule(@(t) x_new(t - T/4) .* exp(-1i * k * 2 * pi * t / T), ll, ul, 100000) / T, k_range);

% Theoretical relation between old coefficients (ak) and new coefficients (ck)
ak_theoretical = @(k) (1 / pi) * (sin(k * pi / 4) - sin(3 * k * pi / 4)) / k;

% Calculate ck coefficients
ck = @(k) (1 / (1i * pi * k)) * (exp(-1i * k * pi / 4) - exp(1i * k * 3 * pi / 4));

% Verify the relation for a3 and c3
a3_idx = find(k_range == 3);
a3 = coeffs_new(a3_idx);
c3 = ck(3);

% Display the verification results
fprintf('a3: %f\n', a3);
fprintf('c3: %f\n', c3);
fprintf('a3_theoretical: %f\n', ak_theoretical(3));

% Plot the shifted function and its Fourier series approximation
figure;
plot(t, x_shifted(t), 'r', 'LineWidth', 2);
hold on;
fourier_series = zeros(size(t));
for k_idx = 1:length(k_range)
    k = k_range(k_idx);
    fourier_series = fourier_series + coeffs(k_idx) * exp(1i * k * 2 * pi * t / T - 1i * k * pi / 2);
end
fourier_series = real(fourier_series);
plot(t, fourier_series, 'b', 'LineWidth', 2);
legend('Shifted Function', 'Fourier Series Approximation', 'Location', 'Best');
xlabel('t');
ylabel('x(t)');
title('Shifted Function and Fourier Series Approximation');


%%% PART 6 %%%

% Step 1: Read the original audio file
[original_audio, Fs] = audioread('original_audio.wav');

% Step 2: Define rotation speed (in radians per second) and time vector
rotation_speed = 0.1; % Adjust this value to control the speed of rotation
t = (0:length(original_audio)-1) / Fs;

% Step 3: Define sinusoidal waves with phase shift for left and right channels
phase_shift_left = 2*pi*rotation_speed*t;
phase_shift_right = -4*pi*rotation_speed*t; % Opposite direction for right channel

sin_wave_left = sin(phase_shift_left);
sin_wave_right = sin(phase_shift_right);

% Step 4: Apply phase shift to original audio for left and right channels
modified_audio_left = original_audio(:, 1) .* sin_wave_left(:);
modified_audio_right = original_audio(:, 2) .* sin_wave_right(:);

% Step 5: Combine channels to create stereo audio
modified_audio = [modified_audio_left, modified_audio_right];

% Step 6: Save the modified audio
audiowrite('360_audio.wav', modified_audio, Fs);

% Display a message
disp('360° audio file has been created successfully.');
