function nest = htNoiseEstimate(x,chanEst,cfgHT,varargin)
%htNoiseEstimate Estimate noise power using HT data field pilots
%
%   NEST = htNoiseEstimate(X,CHANEST,CFGHT) estimates the mean noise power
%   in watts using the demodulated pilot symbols in the HT data field and
%   estimated channel at pilot subcarriers location. The noise estimate is
%   averaged over the number of symbols and receive antennas.
%
%   X is the received time-domain HT-Data field signal. It is a Ns-by-Nr
%   matrix of real or complex values, where Ns represents the number of
%   time-domain samples in the HT-Data field and Nr represents the number
%   of receive antennas.
%
%   CHANEST is a complex Nst-by-(Nsts+Ness)-by-Nr array containing the
%   estimated channel at data and pilot subcarriers, where Nst is the
%   number of subcarriers, Nsts is the number of space-time streams, Ness
%   is the number of extension streams.
%
%   CFGHT is the format configuration object of type <a
%   href="matlab:help('wlanHTConfig')">wlanHTConfig</a>.
%
%   NEST = htNoiseEstimate(...,SYMOFFSET) specifies the sampling offset as
%   a fraction of the cyclic prefix (CP) length for every OFDM symbol, as a
%   double precision, real scalar between 0 and 1, inclusive. The OFDM
%   demodulation is performed based on Nfft samples following the offset
%   position, where Nfft denotes the FFT length. The default value of this
%   property is 0.75, which means the offset is three quarters of the CP
%   length.

%   Copyright 2018-2023 The MathWorks, Inc.

%#codegen

nest = wlan.internal.htNoiseEstimate(x,chanEst,cfgHT,varargin{:});

end