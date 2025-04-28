# Fieldscale C++ Implementation

We provide a minimal C++ version of Fieldscale and simple example.
We hope this version can be included and utilized in any C++ based TIR codebase.
For more details in parameter tuning, please refer to [python](../python/) directory.


## Installation

```bash
sudo apt install libopencv-dev cmake g++
```
Any opencv version greater than 3.4 will work.

## clone & build

```bash
# on cpp dir
mkdir build && cd build
cmake .. && make
```

## run demo

```bash
# on build dir
./fieldscale_demo ../sample/thermal_14bit.png          # writes fieldscaled.png
./fieldscale_demo ../sample/thermal_14bit.png out.png  # custom output name
```