# RTMOT
RTMOT (Real Time Multiple Object Tracking) is a system which can track objects in real time while they are moving in a multi-camera environment and re-identify them when they re-appear in the same or another camera. Concept of the system is an enhancement of TLD of Zdenek Kalal which is a single camera single object tracking system. Due to its high computationally intensive and in order to achieve real time performance we have designed most of the system using CUDA (Computer Unified Device Architecture). Using CUDA we were able to utilize the parallelism capabilities of NVIDIA GPU. Our optimization of the algorithms, careful usage of parallel computing and proper utilization of GPU resources  have contributed in achieving a processing time of less than 60ms for multi objects in multi camera environment.
