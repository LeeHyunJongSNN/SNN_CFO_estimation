function [FER, BER] = calculateOFDMBER(msg, decMsgInBits, numFramesDetected)
% CALCULATEOFDMBER: BER calculation based on the repeatedly transmitted
% message and the detected frames.

% Copyright 2014-2018 The MathWorks, Inc.

% Reconstruct transmitted bit message 
txMsgInBits    = coder.const(double(dec2bin(char(msg), 7).'));
txMsgInBits    = txMsgInBits(:) - 48;
txMsgInBitsAll = repmat(txMsgInBits, 1, numFramesDetected);
bitMsgLen      = length(txMsgInBits);

% Remove padded bits from detected frames
rxMsgInBits    = reshape(decMsgInBits, length(decMsgInBits)/numFramesDetected, numFramesDetected);
rxMsgInBitsAll = rxMsgInBits(1:bitMsgLen, :); 

% FER calculation
numFrmErr = sum(any(txMsgInBitsAll ~= rxMsgInBitsAll));
FER = numFrmErr/numFramesDetected;

% BER calculation
numBitErr = sum(txMsgInBitsAll(:) ~= rxMsgInBitsAll(:));
BER = numBitErr/(bitMsgLen*numFramesDetected);

% [EOF]
