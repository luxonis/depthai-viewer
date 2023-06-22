<h1 align="center">
  <a href="https://www.rerun.io/">
    <img alt="banner" src="https://user-images.githubusercontent.com/1148717/218142418-1d320929-6b7a-486e-8277-fbeef2432529.png">
  </a>
</h1>

<h1 align="center">
  <a href="https://pypi.org/project/depthai-viewer/">                          <img alt="PyPi"           src="https://img.shields.io/pypi/v/depthai-viewer.svg">                              </a>
  <!-- <a href="https://crates.io/crates/rerun">                               <img alt="crates.io"      src="https://img.shields.io/crates/v/rerun.svg">                                </a> -->
  <a href="https://github.com/rerun-io/rerun/blob/master/LICENSE-MIT">    <img alt="MIT"            src="https://img.shields.io/badge/license-MIT-blue.svg">                        </a>
  <a href="https://github.com/rerun-io/rerun/blob/master/LICENSE-APACHE"> <img alt="Apache"         src="https://img.shields.io/badge/license-Apache-blue.svg">                     </a>
  <!-- <a href="https://discord.gg/Gcm8BbTaAj">                                <img alt="Rerun Discord"  src="https://img.shields.io/discord/1062300748202921994?label=Rerun%20Discord"> </a> -->
</h1>

# Depthai Viewer: The visualization tool for DepthAi

<p align="center">
  <img width="800" alt="Rerun Viewer" src="https://user-images.githubusercontent.com/1148717/218763490-f6261ecd-e19e-4520-9b25-446ce1ee6328.png">
</p>

## Getting started

### Prerequisites

- A working version of Python>=3.8
- Libjpeg turbo:
  - MacOS: `brew install jpeg-turbo`
  - Ubuntu: `sudo apt install libturbojpeg`
  - Windows: [libjpeg-turbo official installer](https://sourceforge.net/projects/libjpeg-turbo/files/)

### Install

```sh
# ---------- Linux / MacOS ----------
python3 -m pip install depthai-viewer
# ------------- Windows -------------
python -m pip install depthai-viewer
```
### Run
```sh
python3 -m depthai_viewer
# --------  OR  ---------
depthai-viewer
```



### Documentation
Depthai Viewer can be used as a visualization tool, just like [rerun](https://rerun.io). It uses largely the same python logging api, so you can reefer to the relevant rerun documentation:
- 📚 [High-level docs](http://rerun.io/docs)
- ⚙️ [Examples](examples)
- 🐍 [Python API docs](https://ref.rerun.io/docs/python)
- ⁉️ [Troubleshooting](https://www.rerun.io/docs/getting-started/troubleshooting)

## Status

We are in early beta, however any DepthAi capable device should work with the viewer. If it doesn't, feel free to open an [issue](https://github.com/luxonis/depthai-viewer/issues).
