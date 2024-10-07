# cuda 实战demo 

* [GPU高性能编程CUDA实战](https://hpc.pku.edu.cn/docs/20170829223652566150.pdf)
* 
  * [cuda下载地址](https://developer.nvidia.com/cuda-downloads)，
  * [需要配置环境变量](https://www.ee.torontomu.ca/~courses/ee8218/CUDA%20Instructions.pdf)
  * cmd powershell bash git-bash 检查nvcc.
  ```bash
    nvcc --version
  ```
* [opencv下载](https://opencv.org/releases/)
* vscode 配置  
 launch.json
```json
{
  "version": "0.2.0",
  "configurations": [
      {
          "name": "(Windows) Launch",
          "type": "cppvsdbg",
          "request": "launch",
          "program": "${workspaceFolder}/bin/test.exe",
          "args": [],
          "stopAtEntry": false,
          "cwd": "${workspaceFolder}",
          "environment": [],
          "externalConsole": true
      }
  ]
}
```
tasks.json
```json
{
	"version": "2.0.0",
	"tasks": [
		{
			"type": "shell",
			"label": "build",
			"command": "nvcc",
			"args": [
				"-g", 
				"-o", 
				"./bin/test.exe", 
				"chapter4/julia_gpu.cu", 
				"--include-path", 
				"D:/software/OpenCV/opencv/opencv/build/include",
				"--library",
				"D:/software/OpenCV/opencv/opencv/build/x64/vc16/lib/opencv_world4100"
			],
			"problemMatcher": [],
			"group": {
				"kind": "build",
				"isDefault": true
			}
		}
	]
}
```
```bash
// task build  
ctrl + shift + b
```