Why duplicate?
==============

Without doubt, [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) has fantastic functionality,
performance and the authors have put a lot of thought into how it is designed.

However, I personally feel like the modularity could be more fine grained.
More specifically, have a look at the corresponding [paper](https://arxiv.org/abs/2101.12624).
I don't fully agree with the "Hackability" criterion:
I believe that copy-pasting and changing code ("hacking") is to be minimised by
using good coding practices ([UNIX principle](https://en.wikipedia.org/wiki/Unix_philosophy)).

I do agree that some algorithms have overlap and copy-pasting is faster than writing
from scratch. Julia has magnificient options to avoid boilerplate code in that case
(macros, generated functions, ...).
Hence, "Hackability" should be removed and more importance put on "write code others understand" and
"modularise appropriately" as well as "develop tests for your code".

Applied to [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl),
in my opinion, simulation should be independent of reconstruction.
Similarly, non-Cartesian Fourier reconstruction could be outsourced since it is not exclusively used for MRI.
For example, there is another [Julia package](https://github.com/JuliaApproximation/FastTransforms.jl) for that.
Also, I feel like a general MRI reconstruction package should not be concerned with reading in (vendor-specific) data formats.

My propositions seem to be extra effort for nothing, true.
In that case, please make a self-experiment and keep track of time lost due to poor modularisation and
painful "hacking" into someone else's code.
Then consider that many others in the research community undergo a similar process.
I am convinced that extra effort developers have to make the proposed changes pays off in the long run.

For my research work, I found that asking oneself

1. Into which atomic sub-problems can I divide my specific problem?
2. Which of these are still specific to my problem?

and modularising accordingly doesn't cost too much extra time and leads to pleasant-to-work-with
code both during development and reuse by you and others.

This is only my opinion, please take it with a grain of salt,
[MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) is a great package
and will likely give you what you need ;) .

