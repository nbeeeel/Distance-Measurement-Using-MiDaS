# Distance-Measurement-Using-MiDaS
An effective Deep Learning based distance measurement technique without any calibration. 
![image](https://github.com/nbeeeel/Distance-Measurement-Using-MiDaS/assets/112415272/8c153601-0076-4f26-b6ed-f57859bacf1c)

## Methodology
Here in this project we effectively made use of Deep Learning based depth estimation model MiDaS for estimating the depth value of each pixel.
Along with MiDaS, mediapipe's pose landmark estimation model was also used to create a reference landmark for relative distance calculation.

## Environment Creation 

Open Command Prompt and follow these steps:

### For Python Env
```
python -m venv D:/ENVS/MiDaS
MiDaS\Script\activate
```
### For Conda Env
```
conda env create -f <ENV NAME>
conda activate <ENV NAME>
```
## Install Recommended Packages via requirements.txt file 
```
pip install -r requirements.txt
```
