---
hide:
    - navigation
    - toc
---

<h1 align="center">
    <img src="imgs/logo.png" style="width:450px;height:auto;"/>
</h1>

<p align="center">
  <i><font size=6, color="grey">Digital Signal Processing for Big EEGs</font></i>
</p>


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
](https://github.com/psf/black)

<p align="center"  style="font-size: 20px">
<a href="#key-features">Key Features</a>   |  
<a href="#installation">Installation</a>   |  
<a href="#dependencies">Dependencies</a>   |  
<a href="#documentation">Documentation</a>   |  
<a href="#attribution">Attribution</a>   |  
<a href="#contributions">Contributions</a>   |  
<a href="#issues">Issues</a>   |  
<a href="#license">License</a> |
<a href="#acknowledgements">Acknowledgements</a> 
</p>

<hr>

* **Source Code:**  <a href=https://github.com/mscaudill/openseize
                     target=_blank>https://github.com/mscaudill/opensieze
                    </a>
* **White Paper:** <a href="https://github.com/mscaudill/opensieze">
JOSS LINK</a>

<hr>

# Key Features
Recent innovations in EEG recording technologies make it possible to perform
high channel count recordings at high sampling frequencies spanning many
days. This results in big EEG data sets that are often not addressable to
virtual memory. Worse yet, current digital signal processing (DSP) packages
that rely on Matlab&copy; or Scipy's DSP routines require the data to be
a contiguous in-memory array.  <b><a
href=https://github.com/mscaudill/openseize target=_blank>Openseize</a> is a fully
iterative DSP Python package that can scale to the largest of EEG data
sets.</b> It accomplishes this by storing DSP operations, such as filtering,
as on-the-fly iterables that "produce" DSP results one fragment of the data
at a time. Additionally, Openseize is built using time-tested software
design principles that support extensions while maintaining a simple
interface. Lastly, Openseize <a
href=https://github.com/mscaudill/openseize target=_blank>documentation</a> features
in-depth discussions of iterative DSP processing and its implementation.

<font color='black'>
<ul style="background-color:#DEF5E8;">
  <li>Construct sequences of DSP steps that operate completely 'out-of-core' to
  process data too large to fit into memory.</li>
  <li>DSP pipelines are constructed using a familiar Scipy-like API, so you can
  start quickly without sweating the details.</li>
  <li> Supports processing of data from multiple file types including the popular
  European Data Format (EDF).</li>
  <li>Supports 'masking' to filter data sections by artifacts, behavioral states
  or any externally measured signal(s)/annotation(s).</li>
  <li> Effeciently process large data using the amount of memory <u>you</u>
  choose to use.</li>
  <li> DSP tools currently include a large number of FIR & IIR Filters,
  polyphase decomposition resamplers, and spectral estimation tools for both
  stationary and non-stationary data.</li>
  <li> Built using a developer-friendly object-oriented approach to support
  extensibility.</li>
</ul>
</font>

<hr>

# Installation

For each installation guide below, we **strongly** recommend creating a 
virtual environment. This environment will isolate external dependencies 
that may conflict with packages you already have installed on your system. 
Python comes installed with a virtual environment manager called `venv`. 
Additionally, there are environment managers like `conda` that can check 
for package conflicts when the environment is created or updated. For more
information please see:

* <a href=https://realpython.com/python-virtual-environments-a-primer/
   target=_blank>Python Virtual Enironments</a> 
* <a 
href=https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html target=_blank>Conda Enironments</a> 


### Python Virtual Environment
1. Create your virtual environment, Here we name it `my_venv`. 
```console
$ python -m venv my_venv
```

2. Activate your 'my_venv' environment
```console
$ source my_venv/bin/activate
```

3. Install openseize into your virtual environment
```console
(my_venv)$ python -m pip install openseize
```

### Conda
The `conda` environment manager is more advanced than `venv`. When a `conda`
environment is created, `conda` *simultaneously* looks at all the packages 
to be installed to reduce package conflicts. Having said that, `conda`
and `pip`, the tool used to install from pypi, do not always work well
together. The developers of `conda` recommend installing all possible
packages from conda repositories before installing non-conda
packages using `pip`. To ensure this order of installs, Openseize's source 
code includes an environment configuration file (yml) that will build an
openseize `conda` environment. Once built you can then use `pip` to install
the openseize package into this environment. Here are the steps:

1. Download the openseize environment <a
href=https://github.com/mscaudill/openseize/blob/master/environment.yml 
target=_blank>configuration yaml</a> 


2. Create a conda openseize environment.
```console
$ conda env create --file environment.yml
```

3. Activate the `openseize` environment.
```console
$ conda activate openseize
```

4. Install openseize from pypi into your openseize environment.
```
(openseize)$ pip install openseize
```

### From Source
If you would like to develop Openseize further, you'll need the source code
and all development dependencies. Here are the steps:

1. Create a virtual environment 

2. Get the source code move into the project directory
```
$ git clone https://github.com/mscaudill/openseize.git
```

3. Create an editable install with `pip`.
```
$ pip install -e .
```

<hr>

# Dependencies

Openseize requires <b>Python <span>&#8805;</span> 3.10</b> and has the following dependencies:

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
    <td><a href="https://github.com/rjperez94/pywget"
        target=_blank>pywget</a></td>
    <td>https://pypi.org/project/pywget/</td>
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
Openseize provides both tutorials and extensive reference documentation.

# Attribution

```
Citation from Joss
```

And if you really like Openseize, you can star the <a
href=https://github.com/mscaudill/openseize>repository</a> 
<span>&#11088;</span>!

# Contributions

# Issues

# License
Openseize is licensed under the terms of the 3-Clause BSD License.

# Acknowledgements
**This work was generously supported through the Ting Tsung and Wei Fong Chao 
Foundation.**


