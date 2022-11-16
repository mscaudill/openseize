---
title: 'Openseize: A signal processing package for large EEG datasets in Python'
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
  - name: Department of Neuroscience, Baylor College of Medicine, Houston, TX, USA
    index: 1
  - name: Jan and Dan Duncan Neurological Research Institute at Texas Childrens Hospital, Houston, TX, USA
    index: 2
date: 8 November 2022
bibliography: paper.bib
---

# Summary

Electroencencepholgraphy (EEG) is an indispensable clinical and research
tool used to diagnose neurological disease [@Davies2007; @Noachtar2009; @Tierney2012] and discover brain circuit mechanisms that support sensory, mnemonic and cognitive processing [@Woodman2010; @Nunez2021]. Mechanistically, EEGs are non-stationary time-series that reflect on-going changes in the brain's electromagnetic field. This field is modulated by synchronous changes in the electric potential of millions of individual synapses. To understand these time-series, linear digital signal processing (DSP) tools are routinely used to reduce noise, resample the data, remove artifacts, expose the data's spatio-temporal frequency content, and much more. Openseize is a DSP software package written in pure Python that scales to very large EEG datasets, uses an extensible object-oriented architecture, and provides a familiar Scipy-like API [@Virtanen2020].

# Statement of need

### Scalable

Current DSP software packages [@Delorme2004; @Oostenveld2011; @Tadel2011; @Gramfort2013; @Cole2019] make two critical assumptions. First, they assume that the signals to be analyzed are addressable to virtual memory. Second, they assume that values returned from a DSP process, such as filtering, and all subsequent processes are also addressable to memory. Advances in recording technologies over the past decade are degrading these assumptions. Indeed, thin-film electronics innovations allow for the deposition of a large number of electrode contacts onto a single recording device that can be left implanted for months [@Thongpang2011]. These high-channel count and long-duration recordings pose a serious challenge to imperatively programmed DSP software in which the data is modified as it is passed through functions within a program. 

Openseize takes a functional (declarative) programming strategy that allows for constant and tuneable memory overhead during any DSP processing stage. Specifically, this strategy shuttles iterables (called producers) rather than data between the functions within a program. These memory-efficient producers generate on-the-fly fragments of EEG or DSP processed data. Importantly, producers can be composed into sophisticated  DSP pipelines (\autoref{fig:pipeline}) that yield processed arrays lazily during an iteration protocol such as a for-loop.

![Example DSP pipeline for computing the power spectrum of a large EEG dataset. Each DSP process in the pipeline recieves and returns a producer iterable. At the final stage, the power spectral density (PSD) estimator requests an array from the downsampled producer. This triggers all previous DSP producers to generate a single array.\label{fig:pipeline}](pipeline.png)

A consequence of this functional programming approach is that Openseize's algorithms, in contrast to DSP packages solely relying on Matlab [@Matlab2010] and Scipy [@Virtanen2020], are fully iterative.  Restructuring DSP algorithms to support iterative processing is non-trivial as complex boundary conditions and the need to minimize the number of data transfers between the hard-disk and virtual memory create significant challenges. To meet these, Openseize uses a first-in first-out (FIFO) queue data structure to cache arrays. This caching allows boundary conditions that depend on nearby data to be correctly applied and reduces the number of disk reads. This efficient iterative processing allows Openseize to scale to very large EEG recordings.

### Extensibile

 In addition to scaling to large datasets, Openseize is built using an object-oriented architecture to ensure extensibility. This feature, missing in many currently available DSP packages, is crucial in neuroscience research for two reasons. First, there are many different data file types in-use. Abstract base classes [@GOF] help future developers integrate their file types into Openseize by identifying reusable and required abstract methods needed to create producers that Openseize can work with. Second, DSP operations are strongly interdependent. By identifying and abstracting common methods, the algorithms in Openseize are smaller, more maintainable and above all easier to understand.  \autoref{fig:types} diagrams the currently available DSP methods grouped by their abstract types or module names.

 ![Partial list of DSP classes and methods available in Openseize grouped by abstract type and/or module (gray boxes). Each gray box indicates a point of extensibility either through development of new concrete classes or functions within a module.\label{fig:types}](types.png)

### Intuitive API

 Finally, Openseize has an intuitive application programming interface (API). While under the hood, Openseize is using a functional programming approach, from the end-user's perspective, the calling of its functions are nearly identical to Scipy's DSP call signatures. The only noticeable difference is that DSP processed values in a producer are not immediately returned when the producer is created. Rather, the values are generated when the producer is iterated over. To guide new users, documentation that includes in-depth discussions about DSP algorithms and their implementations in Openseize are included as a series of Jupyter notebooks [@jupyter]. To maintain the clarity and extensibility Openseize's API, graphical user interfaces (GUIs) have been avoided. This decision reflects the fact that many current DSP packages have inconsistent APIs depending on whether the modules are invoked from the command-line or a GUI.   

In summary, Openseize fulfills a need in neuroscience research for DSP tools that can scale with increasing EEG data sizes, are extensible enough to handle new data types and methods, and are accessible to both end-users and developers.

# Acknowledgements

We thank Josh Baker for help in debugging and testing Openseize on real-world EEG data as well as critical reading of the manuscript. This work was generously supported through the Ting Tsung and Wei Fong Chao Foundation.

# References
