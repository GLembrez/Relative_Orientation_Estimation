# Estimation of relative orientations of two rigid bodies with mechanical connection using MIMUs measurements

Given two tigid bodies $(S_1)$ and $(S_2)$ connected with a spherical joint, the aim of this project is to estimate the relative orientation of the two bodies using MIMUs measurements. Let us introduce the main notations.

| symbol | meaning |
| --- | --- |
| $q_t$ | quaternion representing the orientation of $(S_2)$ with respect to $(S_1)$ at time $t$ |
|$\hat{.}$| mean of a variable |
| $a_{i}^{A}$ | acceleration of the center of the joint computed from IMU $i$ |
| $\omega$ | rotation velocity of the joint |
| $m_i$ | measurement from magnetometer $i$ |
| $x$ | state mean |
| $P$ | state covariance |


## MEKF

The state $x \in \mathbb{R}^6$ comprises the quaternion deviation linearized around the quaternion $q_t$ and the bias on the measurement of the joint velocity.

$$
q_t = 
\begin{bmatrix} \cos \| \eta_t \|_2 \\ \frac{\eta_t}{\|\eta_t \|} \sin \| \eta_t \|_2  \end{bmatrix} \odot
\hat{q}_t
$$


## Review of state of the art relative orientation estimation methods

| Title | Date | Observer | Mags | Geom | Comments |
|---|---| --- | --- | --- | --- |
| Sensor fusion algorithms for orientation tracking via magnetic and inertial measurement units: An experimental comparison survey  | 2021 |  LCF <br/> NCF <br/>  LKF <br/> EKF <br/> CKF <br/> SRUKF <br/> SRCKF  | yes | no | absolute orientation only <br/> matlab code is available <br/> sufficient for benchmarking against all the methods without geometric constraints |
| Fast relative sensor orientation estimation in the presence of real-world disturbances | 2021 | MEKF | no | yes | only one citation <br/> code not available <br/> data not available |
| Magnetometer-free Realtime Inertial Motion Tracking by Exploitation of Kinematic Constraints in **2-DoF Joints** | 2019 | optimization based smoothing | no | yes | Uses constraints to replace mags and deactivates the compensation close to singularity (this approach is not general) |
| IMU-Based Joint Angle Measurement for Gait Analysis | 2014 | optimization based smoothing | no | yes | the author says that both acc and gyro estimates can be used in sensor fusion method but rely on a simpler expression (convolution) |
| Observability of the relative motion from inertial data in kinematic chains | 2022 | None | no | yes | formulates the assumptions on the movement neede to guarantee mag-free observability  <br/> **assumes noise and bias free IMUs**|
| Drift-Free Inertial Sensor-Based Joint Kinematicsfor Long-Term Arbitrary Movements | 2020 | MEKF <br/> optimization based smoothing | no | yes | Arbitrary movements do not include adversarial movements <br/> pseudo code is clear |
| A Fast and Robust Algorithm for Orientation Estimation using Inertial Sensors | 2019 | CF | yes | yes | absolute orientation only |