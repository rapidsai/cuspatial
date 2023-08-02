# Building cuProj Documentation

As a prerequisite, a [RAPIDS compatible GPU](https://docs.rapids.ai/install#system-req) is required to build the docs since the notebooks in the
docs execute the code to generate the HTML output.

## Steps to follow:

In order to build the docs, we need the conda dev environment from cuproj to build cuproj from
source.

1. Create a conda env and build cuproj from source. The dependencies to build cuproj from source are
installed in that conda environment, and then cuproj is built and installed into the same environment.

2. Once cuproj is built from source, navigate to `cuspatial/docs/cuproj`. When the documentation
is written run the makefile to build HTML:


```bash
# in the same directory as Makefile
make html
```
This runs Sphinx in the shell, and outputs to `build/html/index.html`.


## View docs web page by opening HTML in browser:

First navigate to `/build/html/` folder, i.e., `cd build/html` and then run the following command:

```bash
python -m http.server
```
Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

```
https://<host IP-Address>:8000
```
Now you can check if your docs edits are formatted correctly and read well.
