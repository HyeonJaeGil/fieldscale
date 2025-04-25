# Fieldscale-CPP (14-bit → 8-bit)

**C++17 port** of the original Python code from  
*“Fieldscale: Locality-Aware Field-Based Adaptive Rescaling for Thermal Infrared Image”*  
(Hyeonjae Gil, Myeon-Hwan Jeon, Ayoung Kim – IEEE RA-L 2024).

This repository keeps the algorithm identical, ships a tiny demo executable, and can be dropped into ROS or any C++ project.

---

# 1  Dependencies

| Library | Tested version | Ubuntu/Debian install |
|---------|----------------|-----------------------|
| OpenCV (core, imgproc, imgcodecs, highgui) | ≥ 3.4 (4.x used) | `sudo apt install libopencv-dev` |
| CMake | ≥ 3.10 | `sudo apt install cmake` |
| C++17 compiler | GCC 9 / Clang 12 / MSVC 2022 | `sudo apt install g++` |

---

# 2  Build & run the demo



## clone & build

```bash
git clone -b cpp-port --single-branch https://github.com/HyeonJaeGil/fieldscale.git
cd fieldscale/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## run on a 14-bit image

```bash
./fieldscale_demo ../sample/thermal_14bit.png          # writes fieldscaled.png
./fieldscale_demo ../sample/thermal_14bit.png out.png  # custom output name
```