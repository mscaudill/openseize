<h1 align="center">
  <a href="https://github.com/mscaudill/opensieze">
    <img src="https://github.com/mscaudill/openseize/blob/master/core/imgs/logo2.png" alt="openseize_logo" height="300">
  </a>
</h1>

<p align="center">
  <b>This repository contains the Openseize source code, an iterative digital signal processing package for big EEG data in Python. </b>
</p>

<p align="center">
<a href="#introduction">Introduction</a>   |  
<a href="#installation">Installation</a>   |  
<a href="#dependencies">Dependencies</a> |   |  
<a href="#documentation">Documentation</a>   |  
<a href="#attribution">Attribution</a>   |  
<a href="#contributions">Contributions</a>   |  
<a href="#issues">Issues</a>
</p>

# Introduction

Recent innovations in EEG recording technologies make it possible to perform high channel density recordings over several months at high sampling frequencies. This results in big EEG data sets that are often not addressable to virtual memory. Worse yet, current digital signal processing (DSP) packages that rely on Matlab &c or Scipy's DSP routines require the data to be a contiguous in-memory array. Openseize is a fully iterative DSP Python package that can scale to the largest of EEG data sets. It accomplishes this by storing DSP operations, such as filtering, as on-the-fly iterables that "produce" DSP results one fragment of the data at a time. Additionally, Openseize is built using time-tested software design principles that support extensions while maintaining a simple interface.

# Installation

## pypi

## conda-forge

## from source

# Dependencies

Openseize requires <b>Python <span>&#8805;</span> 3.6</b> and has the following dependencies:

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

# Documentation

# Attribution

If you have used Openseize in a publication please cite it using:</br>

```
Citation from Joss
```

And if you really like Openseize, you can star this repository <span>&#11088;</span>

# Contributions

# Issues
