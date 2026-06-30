# miniaudio

Vendored from `https://github.com/mackron/miniaudio`.

- File: `miniaudio.h`
- Version in header: `0.11.25`
- License: choice of public domain / Unlicense or MIT-0; the license text is embedded at the end of `miniaudio.h`.

This dependency is available for future C++ audio file I/O and server/CLI work. The current Higgs reference-audio preprocessing still uses the local torchaudio-matched sinc resampler path because miniaudio's built-in resampler is not equivalent to torchaudio's default `sinc_interp_hann` contract that the parity gates depend on.
