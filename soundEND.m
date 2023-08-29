sf = 22050;                        % sample frequency (Hz)
d = 0.4;                           % durati0n (s)
n = sf * d;                        % number of samples

% set carrier
cf = 1000;                         % carrier frequency (Hz)
c = (1:n) / sf;                    % carrier data preparation
c = sin(2 * pi * cf * c);          % sinusoidal modulation

% set modulator
mf = 5;                            % modulator frequency (Hz)
mi = 0.5;                          % modulator index
m = (1:n) / sf;                    % modulator data preparation
m = 1 + mi * sin(2 * pi * mf * m); % sinusoidal modulation

% amplitude modulation
s = m .* c;                        % amplitude modulation

% sound presentation
sound(s, sf);                      % sound presentation
pause(d + 0.5);                    % waiting for sound end