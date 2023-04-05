--- 
title: "Openseize: A digital signal processing package for large EEG
datasets in Python" 
tags:
  - Python
  - EEG
  - neuroscience
  - signal processing
  - local field potentials
  - time series analysis
authors:
  - name: Matthew S. Caudill 
    orcid: 0000-0002-3656-9261 
    affiliation: "1, 2" # (multiple affiliations must be quoted) 
affiliations:
  - name: Department of Neuroscience, Baylor College of Medicine, Houston,
      TX, United States of America
    index: 1
  - name: Jan and Dan Duncan Neurological Research Institute at Texas
      Childrens Hospital, Houston, TX, United States of America 
    index: 2 
date: 8 November 2022
bibliography: paper.bib
---

# Summary

Electroencephalography (EEG) is an indispensable clinical and research tool
used to diagnose neurological disease [@Davies2007; @Noachtar2009;
@Tierney2012] and discover brain circuit mechanisms that support sensory,
mnemonic and cognitive processing [@Woodman2010; @Nunez2021].
Mechanistically, EEGs are non-stationary time-series that capture
alterations in the brain's electromagnetic field arising from synchronous
synaptic potential changes across neuronal populations. Linear digital
signal processing (DSP) tools are routinely used in EEGs to reduce noise,
resample the data, remove artifacts, expose the data's spatio-temporal
frequency content, and much more. Openseize is a DSP package written in
pure Python that scales to very large EEG datasets, employs an extensible
object-oriented architecture, and provides a familiar Scipy-like API
[@Virtanen2020].

# Statement of need

### Scalable

Current DSP software packages [@Delorme2004; @Oostenveld2011; @Tadel2011;
@Gramfort2013; @Cole2019] make two critical assumptions. First, that the
signals to be analyzed are addressable to virtual memory. Second, that the
values returned from a DSP process, such as filtering, and all subsequent
processes are likewise addressable to memory. Advances in recording
technologies over the past decade are degrading these assumptions. Indeed,
thin-film electronics innovations allow for the deposition of a large number
of electrode contacts onto a single recording device that can be left
implanted for months [@Thongpang2011]. These high-channel count
long-duration recordings pose a serious challenge to imperatively programmed
DSP software in which data is stored as it passes through and between
functions within a program.

Openseize takes a declarative programming approach that allows for constant
and tuneable memory overhead.  Specifically, this approach shuttles
iterables (called producers) rather than data between the functions within
a program. These memory-efficient producers generate on-the-fly fragments
of processed data. Importantly, all of Openseize's functions and methods
accept and return producers. This feature allows for the composition of DSP
functions into iterative processing pipelines (\autoref{fig:pipeline}) that
yield processed data lazily during an iteration protocol such as a for-loop.

![Example DSP pipeline for computing the power spectrum of a large EEG
dataset. Each DSP process in the pipeline receives and returns a producer
iterable. At the final stage, the power spectral density (PSD) estimator
requests an array from the downsampled producer triggering all previous
producers to generate a single array.\label{fig:pipeline}](pipeline.png)

A consequence of this functional programming approach is that DSP pipelines
in Openseize, in contrast to pipelines relying on Matlab [@Matlab2010] and
Scipy [@Virtanen2020], are fully iterative. Restructuring DSP algorithms to
support iterative processing is non-trivial because complex boundary conditions
and the need to minimize data transfers between disks and virtual
memory creates significant challenges. To meet these, Openseize uses
a first-in first-out (FIFO) queue data structure to cache arrays. FIFO
caching allows previously seen data to influence the boundary conditions of
in-process data and reduces the number of disk reads. This data structure in
combination with both producers and iterative algorithms allows Openseize to
scale to massive data recordings.

### Extensibile

In addition to its scalability, Openseize employs an extensible
object-oriented architecture. This feature, missing in many currently
available DSP packages, is crucial in neuroscience research for two reasons.
First, there are many different data file types in-use. Abstract base
classes [@GOF] help future developers integrate their file types into
Openseize by identifying required methods needed to create producers that
Openseize's algorithms can process. Second, DSP operations are strongly
interdependent. By identifying and abstracting common methods, the
algorithms in Openseize are smaller, more maintainable and above all, easier
to understand.  \autoref{fig:types} diagrams the currently available DSP
methods grouped by their abstract types or module names.

 ![Partial list of DSP classes and methods available in Openseize grouped by
abstract type and/or module (gray boxes). Each gray box indicates a point of
extensibility either through development of new concrete classes or
functions within a module.\label{fig:types}](types.png)

### Intuitive API

Finally, Openseize has an intuitive application programming interface (API).
While under the hood, Openseize is using a declarative programming approach,
from the end-user's perspective, the calling of its functions are similar
to Scipy's DSP call signatures. The main difference is that producers do not
return DSP processed values when created. Rather, the values are generated
when the producer is iterated over. To help new users understand the
implications of this, Openseize includes extensive in-depth discussions
about DSP algorithms and their iterative implementations in a series of
Jupyter notebooks [@jupyter]. Importantly, to maintain the clarity and
extensibility of Openseize's API, graphical user interfaces (GUIs) have been
avoided. This decision reflects the fact that many current DSP packages have
inconsistent APIs depending on whether the modules are invoked from the
command-line or a GUI.   

In summary, Openseize fulfills a need in neuroscience research for DSP tools
that scale to large EEG recordings, are extensible enough to handle new
data types and methods, and are accessible to both end-users and
developers.

# Acknowledgements

We thank Josh Baker for help in debugging and testing Openseize on
real-world EEG data as well as critical reading of the manuscript. This work
was generously supported through the Ting Tsung and Wei Fong Chao
Foundation.

# References
