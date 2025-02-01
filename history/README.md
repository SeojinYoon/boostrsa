
- 20240201
	- cupy compatibility error with cuda
		- On CUDA 12.2 or later CUDA Runtime header files are required to compile kernels
		- NVRTC_ERROR_COMPILATION
		- Solution
			- pip install cupy-cuda12x
                        - pip install "nvidia-cuda-runtime-cu12==12.6.*"


