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
<a href="#acknowledgements">Acknowledgements</a> |
<a href="#license">License</a>
</p>


<hr>

* **Source Code:**  <a href="https://github.com/mscaudill/opensieze">
(https://github.com/mscaudill/opensieze)
</a>
* **White Paper:**

<hr>

# Key Features
Recent innovations in EEG recording technologies make it possible to perform
high channel count recordings at high sampling frequencies spanning many
days. This results in big EEG data sets that are often not addressable to
virtual memory. Worse yet, current digital signal processing (DSP) packages
that rely on Matlab&copy; or Scipy's DSP routines require the data to be
a contiguous in-memory array. <b><a
href=https://github.com/mscaudill/openseize>Openseize</a> is a fully
iterative DSP Python package that can scale to the largest of EEG data
sets.</b> It accomplishes this by storing DSP operations, such as filtering,
as on-the-fly iterables that "produce" DSP results one fragment of the data
at a time. Additionally, Openseize is built using time-tested software
design principles that support extensions while maintaining a simple
interface. Lastly, the Openseize <a
href=https://github.com/mscaudill/openseize>documentation</a> features
in-depth discussions of iterative DSP processing and its implementation.

* Iteratively produce data from a variety of data sources including the
  popular European Data Format binary file type.
* Build iterative data producers that can *mask* artifacts or select time
  periods corresponding to behavioral states.
* Iteratively filter producers of data with a selection of Finite Impulse 
  Response (FIR) and Infinite Impulse Response (IIR) filters.
* Resample producers of data with effecient polyphase decompositions.
* Measure a signals frequency content with Power spectrums and Short-time
  Fourier transforms. 

# Installation

For each installation guide below, we **strongly** recommend creating a 
virtual environment. This environment will isolate external dependencies 
that mayconflict with packages you already have installed on your system. 
Python comes installed with a virtual environment manager called `venv`. 
Additionally, there are environment managers like `conda` that can check 
for package conflicts when the environment is created or updated. For more
information please see:

* [Python virtual environments
](https://realpython.com/python-virtual-environments-a-primer/)
* [Conda environments
](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Python Virtual Environment
1. Create your virtual environment, Here we name it `my_venv` 
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

### From Source

## Dependencies

Openseize requires <b>Python <span>&#8805;</span> 3.10</b> and has the following dependencies:

<table>

<tr>
    <th>package</th>
    <th>pypi</th>
    <th>conda</th>
  </tr>

<tr>
    <td><a href="https://requests.readthedocs.io/en/latest/">requests</a></td>
    <td>https://pypi.org/project/requests/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://github.com/rjperez94/pywget">pywget</a></td>
    <td>https://pypi.org/project/pywget/</td>
    <td></td>
  </tr>

<tr>
    <td><a href="https://numpy.org/doc/stable/index.html#">numpy</a></td>
    <td>https://pypi.org/project/numpy/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://https://scipy.org/">scipy</a></td>
    <td>https://pypi.org/project/scipy/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://matplotlib.org/">matplotlib</a></td>
    <td>https://pypi.org/project/matplotlib/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href="https://ipython.org/">ipython</a></td>
    <td>https://pypi.org/project/ipython/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href=https://jupyter.org/>notebook</a></td>
    <td>https://pypi.org/project/jupyter/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href=https://docs.pytest.org/>pytest</a></td>
    <td>https://pypi.org/project/pytest/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

<tr>
    <td><a href=https://psutil.readthedocs.io/en/latest/>psutil</a></td>
    <td>https://pypi.org/project/psutil/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

</table>

# Contributions

# Acknowledgements
This work was generously supported through the Ting Tsung and Wei Fong Chao 
Foundation.

# License
Openseize is licensed under the terms of the 3-Clause BSD License.


