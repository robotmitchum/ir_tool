# Deconvolution example/test files

## Introduction

The Playstation SPU features a lo-fi reverberation effect

These audio files were made by recording the reverb presets from the PS1 CD Player using an emulator

The PS1 CD player has 4 reverberation presets :
* Dome
* Hall
* Church
* Studio

## Content of the directory

* "click.wav" is an impulse used as a cue for the start of the recording
* "sweep.wav" is a logarithmic sweep tone

Both were generated using IR tool

* "{preset}_click.wav" is the direct response from "click.wav"
* "{preset}_sweep.wav" is the response from "sweep.wav"
* "{preset}_sweep_deconvolved.wav" is the result of the deconvolution

## Settings

The following settings were used for the deconvolution :

* ☑ Deconvolve
* ☑ Trim End -99db
* ☑ Fade Out -90db 
* ☑ Normalize "compensate"