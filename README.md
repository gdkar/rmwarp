# DESIGN

librmwarp ( reassignment method warp ) is a vocoder type timescale modification
library based on the mothod of time-frequency reassignment.

in the following, the function `stretch(t)` will be used to represnt the time
remapping function, and `unstretch(t)` will be used to represent `stretch^{-1}(t)`;

at each analysis timestamp, we compute an ( augmented ) reassigned spectrogram,
providing:

*   the transform of the windowed signal.
*   the complex logarithm of the windowed signal ( that is, log-magnitude and phase. )
*   the partial derivative with respect to time of the log spectrum.
*   the partial derivative with respect to frequency of log spectrum.
*   the mixed partial derivative of phase.

( and, yeah, we could trivially also compute the mixed partial of log-magnitude, but
i haven't yet thought of a reason that that'd ~possibly be actually useful. similarly,
i'll maybe / probably end up dropping computation of the frequency-derivative of
log-magnitude as well unless that actually starts being used somewhere, since the
spectrum state is fairly large. )

this gives us access to

*   time reassignment factor, i.e. "local group delay."
*   frequency reassignment factor, i.e., the unwrapped phase difference term from a
    conventional phase vocoder, only without mod 2 * M_PI ambiguity
*   a measure of whether particular frequency bin, in a particular analysis window,
    is "sinusoid like" or "impulsive transient like"

also, since we have the ( exact ) time derivative of the log magnitude, we can perform
cubic interpolation in log-domain, and then convert back to magnitude, which should give
( hopefully ) appreciably better accuracy than just performing linear interpolation on
magnitude.

to get effective phase at intermediate points, we just integrate `dPhi(unstretch(t))`
from the previous synthesis time point to the desired time.

probably we'll want to just do this using trapazoid rule. in theory, we could maybe
improve on that by since we have the actual ( pre-stretching ) phase values, as well
as the derivatives. the approach there would be to, e.g., compute the hermite
interpolating polynomial for unstretched phase between the adjacent pair of analysis
frames, compute its derivative symbolically, and then integrate *that* over mapped
over the desired time range.

since in practice the second derivative of phase is
probably in general not constant, this should give a more accurate result. on the
other hand, the thing we're *starting* with is already a substantially better fit
than the actually-just-finite-differences-and-then-scale approach used in a normal
phase vocoder type system, and honestly normal phase vocoders are pretty usable,
it seems unlikely the improved phase accuracy ( if even meaningful ) would make
a real difference to the output.

the mixed partial derivative of phase gives a measure of "impulsiveness" for fft
bins and, for bins that are "impulsive," the local group delay then gives an estimate
of the effective time of the transient ( or, really, of when the local frequency
components would have all been in-phase, which is approximately the same. ) we can
use that pretty directly to decide when to phase-reset parts of the spectrum. this
comes with a couple caveats:

*   if we decide that a given fft bin should be phase-reset in a given synthesis
    frame, we probably will want to collect the surrounding bins that correspond
    to the same spectral peak, and reset those as well. ( obviously )
*   if we end up with multiple close, but not contiguous, regions undergoing
    phase reset, we probably want to also go ahead and lump in whatever's between
    them. ( sort of by definition, whatever's between them will probably be pretty
    low amplitude, so this partially also falls into the "reset on silence"
    category. )
*   when doing phase-reset on region of influence of a spectral peak, we should
    compute the effective unscaled time of the transient, `t_u`, ( i.e.,
    analysis time +local group delay, ), compute the reassigned phase, corrected
    to be phase at `t_u`, and then perform phase reset to that corrected phase at
    `t_s = stretch(t_u)`. we probably want to use the same `t_u` ( taken from the
    peak bin, or an average, or something ) for all bins corresponding to the same
    fft peak bin.
*   actually, we want to do ( at least ) one better: if we know `t_u` and `t_s` for
    the nearest impulsive transient to a given synthesis time / fft bin, we may
    observe that without correction, the same transient may appear in the synthesis
    spectra of a time range greater than it did in the original. a single transient
    can only effect analysis spectra covering a total duration of ~2 analysis windows,
    but when time-stretching naively it will impact a duration that's longer by the
    time scaling factor.

    to correct for this, when approaching a transient, we should
    stop advancing the interpolated analysis time before the first analysis instant
    containing `t_u`, and then continue using those parameters until the first
    synthesis frame containing `t_s`, at which point we should advance the effective
    analysis time up to correspond.

    it's probable that we also really want to do a similar thing post-transient  (i.e.,
    advance effective analysis time at an accelerated rate until the analysis frame
    is past the transient, and then pause or slow down the advancement of the analysis
    point until the synthesis time catches up. ) i'll.... worry about that later,
    though.

# References

[1]: https://arxiv.org/pdf/0903.3080.pdf "A Unified Theory of Time-Frequency Reassignment"
[2]: http://www.acousticslab.org/learnmoresra/files/fulopfitz2007jasa121.pdf "Separation of Components from Impulses in Reassigned Spectrograms"
[3]: https://pdfs.semanticscholar.org/2042/3dffa92efd5371489e6b11b22779b0a2fc85.pdf "On Phase-Magnitude Relationships in the Short-Time Fourier Transform"
[4]: http://www.mirlab.org/conference_papers/International_Conference/ISMIR%202008/papers/ISMIR2008_174.pdf "Beat Tracking Using Group Delay Based Onset Detection"
[5]: http://recherche.ircam.fr/equipes/analyse-synthese/peeters/ARTICLES/Peeters_2009_DAFX_beat.pdf "Beat-Tracking Using a Probobalistic Framework and Linear Discriminant Analysis"
[6]: https://pdfs.semanticscholar.org/4043/c4b2cea7538728abb8f7934055876de4bc73.pdf "Template-Based Estimation of Time-Varying Tempo"
w
[7]: http://isdl.ee.washington.edu/people/stevenschimmel/publications/SchimmelFitzAtlas_ICASSP2006.pdf "Frequency Reassignment for Coherent Modulation Filtering"
[8]: https://www.researchgate.net/publication/2653258_Extraction_Of_Spectral_Peak_Parameters_Using_A_Short-Time_Fourier_Transform_Modeling_And_No_Sidelobe_Windows "Extraction of Spectral Peak Parameters Using a Short-Time Fourier Transform Modeling and No Sidelobe Windows"
[9]: https://www.researchgate.net/publication/2995027_On_the_Use_of_Windows_for_Harmonic_Analysis_With_the_Discrete_Fourier_Transform "On the Use Of Windows for Harmonic Analysis with the Discrete Fourier Transform"

