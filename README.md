<h1 align="center">
    <img src="https://github.com/mscaudill/openseize/blob/master/docs/imgs/logo.png" 
    style="width:450px;height:auto;"/>
</h1>

<h2 align="center">
  <i><font color='gray'>Digital Signal Processing for Big EEGs</font></i>
</h2>

<p align="center">
  <a href="https://joss.theoj.org/papers/f737f4eb377a3bed6602dac51f6b13b4"><img 
    src="https://joss.theoj.org/papers/f737f4eb377a3bed6602dac51f6b13b4/status.svg" 
    alt="JOSS Review" />
  </a>
  <a href="https://github.com/mscaudill/openseize/blob/master/LICENSE"><img
    src="https://img.shields.io/badge/License-BSD%203--Clause-teal" 
    alt="Openseize is released under the BSD 3-Clause license." />
  </a>
  <a href="https://pypi.org/project/openseize/"><img 
    src="https://img.shields.io/pypi/v/openseize?color=78437E&logo=pypi&logoColor=white" 
    alt="Openseize pypi release" />
  </a>
  <a href="https://github.com/mscaudill/openseize/tree/master#Dependencies"><img 
    src="https://img.shields.io/pypi/pyversions/openseize?logo=python&logoColor=gold" 
    alt="Python versions supported." />
  </a>
  <a href="https://github.com/mscaudill/openseize/actions/workflows/test.yml"><img 
    src="https://img.shields.io/github/actions/workflow/status/mscaudill/openseize/test.yml?label=CI&logo=github" 
    alt="Openseize's test status" />
  </a>
 <a href="https://github.com/mscaudill/openseize/pulls"><img 
    src="https://img.shields.io/badge/PRs-welcome-F8A3A3"
    alt="Pull Request Welcomed!" />
  </a>
</p>


<p align="center"  style="font-size: 20px">
<a href="#Key-Features">Key Features</a>   |  
<a href="#Installation">Installation</a>   |  
<a href="#Dependencies">Dependencies</a>   |  
<a href="#Documentation">Documentation</a>   |  
<a href="#Attribution">Attribution</a>   |  
<a href="#Contributions">Contributions</a>   |  
<a href="#Issues">Issues</a>   |  
<a href="#License">License</a> |
<a href="#Acknowledgements">Acknowledgements</a> 
</p>

<hr>

* **Source Code:**  <a href=https://github.com/mscaudill/openseize
                     target=_blank>https://github.com/mscaudill/opensieze
                    </a>
* **White Paper:** <a href="https://github.com/mscaudill/opensieze">
LINK</a>

<hr>

# Key Features

Recent innovations in EEG recording technologies make it possible to perform
high channel count recordings at high sampling frequencies spanning many
days. This results in big EEG data sets that are often not addressable to
virtual memory. Worse yet, current digital signal processing (DSP)
packages that rely on Matlab&copy; or Scipy's DSP routines require the data
to be a contiguous in-memory array. <b><a
href=https://github.com/mscaudill/openseize target=_blank>Openseize</a> is
a fully iterative DSP Python package that can scale to the largest of EEG
data sets.</b> It accomplishes this by storing DSP operations, such as
filtering, as on-the-fly iterables that "produce" DSP results one fragment
of the data at a time. Additionally, Openseize is built using time-tested
software design principles that support extensions while maintaining
a simple interface. Finally, Openseize's <a
href=https://mscaudill.github.io/openseize/ target=_blank>documentation</a>
features in-depth discussions of iterative DSP processing and its
implementation.

<font color='black'>
<ul style="background-color:#DEF5E8;">
  <li>Construct sequences of DSP steps that operate completely 'out-of-core' 
  to process data too large to fit into memory.</li>
  <li>DSP pipelines are constructed using a familiar Scipy-like API, so you 
  can start quickly without sweating the details.</li>
  <li> Supports processing of data from multiple file types including the 
  popular European Data Format (EDF).</li>
  <li>Supports 'masking' to filter data sections by artifacts, behavioral 
  states or any externally measured signals or annotations.</li>
  <li> Efficiently process large data using the amount of memory <u>you</u>
  choose to use.</li>
  <li> DSP tools currently include a large number of FIR & IIR Filters,
  polyphase decomposition resamplers, and spectral estimation tools for both
  stationary and non-stationary data.</li>
  <li> Built using a developer-friendly object-oriented approach to support
  extensibility.</li>
</ul>
</font>

# Installation

For each installation guide below, we **strongly** recommend creating a 
virtual environment. This environment will isolate external dependencies 
that may conflict with packages you already have installed on your system. 
Python comes installed with a virtual environment manager called `venv`. 
Additionally, there are environment managers like `conda` that can check 
for package conflicts when the environment is created or updated. For more
information please see:

* <a href=https://realpython.com/python-virtual-environments-a-primer/
   target=_blank>Python Virtual Environments</a> 
* <a 
href=https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html target=_blank>Conda Environments</a> 


### Python Virtual Environment

1. Create your virtual environment, Here we name it `my_venv`. 
```Shell
$ python3 -m venv my_venv
```

2. Activate your 'my_venv' environment
```Shell
$ source my_venv/bin/activate
```

3. Install openseize into your virtual environment
```Shell
(my_venv)$ pip install openseize
```

### Conda

The `conda` environment manager is more advanced than `venv`. When a `conda`
environment is updated, `conda` *simultaneously* looks at all the packages
to be installed to reduce package conflicts. Having said that, `conda` and
`pip`, the tool used to install Openseize from pypi, do not always work
well together. The developers of `conda` recommend installing all possible
packages from conda repositories before installing non-conda packages using
`pip`. To ensure this order of installs, Openseize's source code includes an
environment configuration file (yml) that will build an openseize `conda`
environment. Once built you can then use `pip` to install the openseize
package into this environment. Here are the steps:

1. Download the openseize environment <a
href=https://github.com/mscaudill/openseize/blob/master/environment.yml 
target=_blank>configuration yaml</a> 


2. Create a conda openseize environment.
```Shell
$ conda env create --file environment.yml
```

3. Activate the `openseize` environment.
```Shell
$ conda activate openseize
```

4. Install openseize from pypi into your openseize environment.
```Shell
(openseize)$ pip install openseize
```

### From Source

If you would like to develop Openseize further, you'll need the source code
and all development dependencies. Here are the steps:

1. Create a virtual environment with latest pip version.
```Shell
$ python3 -m venv env
$ source env/bin/activate
$ pip install --upgrade pip
```

2. Get the source code
```Shell
$ git clone https://github.com/mscaudill/openseize.git
```

3. CD into the directory containing the pyproject.toml and create an 
editable install with `pip`
```Shell
$ pip install -e .[dev]
```

# Dependencies

Openseize requires <b>Python <span>&#8805;</span> 3.8</b> and has the
following dependencies:

<table>

<tr>
    <th>package</th>
    <th>pypi</th>
    <th>conda</th>
  </tr>

<tr>
    <td><a href="https://requests.readthedocs.io/en/latest/" 
        target=_blank>requests</a></td>
    <td>https://pypi.org/project/requests/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td>wget</td>
    <td>https://pypi.org/project/wget/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://numpy.org/doc/stable/index.html#" 
        target=_blank>numpy</a></td>
    <td>https://pypi.org/project/numpy/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://scipy.org/" 
        target=_blank>scipy</a></td>
    <td>https://pypi.org/project/scipy/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://matplotlib.org/" 
        target=_blank>matplotlib</a></td>
    <td>https://pypi.org/project/matplotlib/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://ipython.org/" 
        target=_blank>ipython</a></td>
    <td>https://pypi.org/project/ipython/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href=https://jupyter.org/ 
        target=_blank>notebook</a></td>
    <td>https://pypi.org/project/jupyter/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href=https://docs.pytest.org/ 
        target=_blank>pytest</a></td>
    <td>https://pypi.org/project/pytest/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href=https://psutil.readthedocs.io/en/latest/ 
        target=_blank>psutil</a></td>
    <td>https://pypi.org/project/psutil/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

</table>

# Documentation

Openseize documentation site has a [quickstart guide](
https://mscaudill.github.io/openseize/quickstart/), [extensive tutorials](
https://mscaudill.github.io/openseize/tutorials/producers/) and [
reference pages](https://mscaudill.github.io/openseize/producer/producer/)
for all publicly available modules, classes and functions.

# Attribution

```
Citation to be added
```

And if you really like Openseize, you can star the <a
href=https://github.com/mscaudill/openseize>repository</a> 
<span>&#11088;</span>!

# Contributions

Contributions are what makes open-source fun and we would love for you to
contribute. Please check out our [contribution guide](
https://github.com/mscaudill/openseize/blob/master/.github/CONTRIBUTING.md)
to get started.

# Issues

Openseize provides custom issue templates for filing bugs, requesting
feature enhancements, suggesting documentation changes, or just asking
questions. *Ready to discuss?* File an issue <a
href=https://github.com/mscaudill/openseize/issues/new/choose>here</a>. 

# License

Openseize is licensed under the terms of the 3-Clause BSD License.

# Acknowledgements

**This work was generously supported through the Ting Tsung and Wei Fong Chao 
Foundation.**



