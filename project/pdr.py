'''
实验前提：采集数据时手机x轴与重力方向垂直

1.Model
参数列表（3个参数）：
线性加速度矩阵（x轴加速度、y轴加速度、z轴加速度）；
重力加速度矩阵（x轴重力加速度、y轴重力加速度、z轴重力加速度）;
四元数矩阵（四元数x、四元数y、四元数z、四元数w）

2.Model参数类型：
numpy.ndarray
'''

import math
import mlp
import numpy as np
from scipy.stats import norm
from scipy import interpolate
from math import degrees, radians, fmod, atan2,sqrt,pi
from copy import deepcopy




class Model(object):
    def __init__(self, li, gr, ma,gy,input,time):
        self.linear = li
        self.gravity = gr
        self.magnetometer = ma
        self.gyroscope = gy #记录陀螺仪数据
        self.input = input  #0:纬度，1：经度，2：方向
        self.time = time
        self.q_est = self.Quaternion(False,1,0,0,0)
        self.mlp_model = mlp.Mlp_Model()

    '''
        用accelerometer 和 gyroscope 计算四元数（quaternion）
    '''
    class Quaternion():
        def __init__(self,empty=True,*args) -> None:
            if empty:
                self.q1 = 0
                self.q2 = 0
                self.q3 = 0
                self.q4 = 0
            
            else:
                assert len(args) == 4 #make sure input right
                self.q1 = args[0]
                self.q2 = args[1]
                self.q3 = args[2]
                self.q4 = args[3]

        def quat_scalar(self,scalar):
            self.q1 *= scalar
            self.q2 *= scalar
            self.q3 *= scalar
            self.q4 *= scalar

        def quat_conjugate(self):
            self.q2 = -self.q2
            self.q3 = -self.q3
            self.q4 = -self.q4

        def quat_Norm(self):
            return sqrt(self.q1 **2 + self.q2 **2 + self.q3 **2 + self.q4 **2)

        def quat_Normalization(self):
            norm = self.quat_Norm()
            self.q1 /= norm
            self.q2 /= norm
            self.q3 /= norm
            self.q4 /= norm


    def quat_mult(self,L,R):
        product = self.Quaternion()

        product.q1 = (L.q1 * R.q1) - (L.q2 * R.q2) - (L.q3 * R.q3) - (L.q4 * R.q4)
        product.q2 = (L.q1 * R.q2) + (L.q2 * R.q1) + (L.q3 * R.q4) - (L.q4 * R.q3)
        product.q3 = (L.q1 * R.q3) - (L.q2 * R.q4) + (L.q3 * R.q1) + (L.q4 * R.q2)
        product.q4 = (L.q1 * R.q4) + (L.q2 * R.q3) - (L.q3 * R.q2) + (L.q4 * R.q1)

        return product

    def quat_sub(self,L,R):
        ans = self.Quaternion()
        ans.q1 = L.q1 - R.q1
        ans.q2 = L.q2 - R.q2
        ans.q3 = L.q3 - R.q3
        ans.q4 = L.q4 - R.q4
        return ans

    def quat_add(self,L,R):
        ans = self.Quaternion()
        ans.q1 = L.q1 + R.q1
        ans.q2 = L.q2 + R.q2
        ans.q3 = L.q3 + R.q3
        ans.q4 = L.q4 + R.q4
        return ans

    def imu_filter(self,ax,ay,az,gx,gy,gz):
        '''a filter of imu, calculate the 四元数'''
        #some date that will be used later
        gyro_mean_error = pi* (5.0/180.0)
        beta = sqrt(3.0/4.0) * gyro_mean_error
        delta_t = 0.01 #100HZ sampling frequency

        q_est_prev = deepcopy(self.q_est)
        q_est_dot = self.Quaternion()
        q_a = self.Quaternion(False,0,ax,ay,az)

        F_g = np.zeros((3),dtype=np.float32) #float
        J_g = np.zeros((3,4),dtype=np.float32) #float

        gradient = self.Quaternion()

        q_w = self.Quaternion(False,0,gx,gy,gz)
        q_w.quat_scalar(0.5)
        q_w = self.quat_mult(q_est_prev,q_w)

        q_a.quat_Normalization()

        F_g[0] = 2*(q_est_prev.q2 * q_est_prev.q4 - q_est_prev.q1 * q_est_prev.q3) - q_a.q2;
        F_g[1] = 2*(q_est_prev.q1 * q_est_prev.q2 + q_est_prev.q3* q_est_prev.q4) - q_a.q3;
        F_g[2] = 2*(0.5 - q_est_prev.q2 * q_est_prev.q2 - q_est_prev.q3 * q_est_prev.q3) - q_a.q4;

        #Compute the Jacobian matrix, for gravity
        J_g[0][0] = -2 * q_est_prev.q3
        J_g[0][1] =  2 * q_est_prev.q4
        J_g[0][2] = -2 * q_est_prev.q1
        J_g[0][3] =  2 * q_est_prev.q2
    
        J_g[1][0] = 2 * q_est_prev.q2
        J_g[1][1] = 2 * q_est_prev.q1
        J_g[1][2] = 2 * q_est_prev.q4
        J_g[1][3] = 2 * q_est_prev.q3
    
        J_g[2][0] = 0
        J_g[2][1] = -4 * q_est_prev.q2
        J_g[2][2] = -4 * q_est_prev.q3
        J_g[2][3] = 0
    
        #now computer the gradient, gradient = J_g'*F_g
        gradient.q1 = J_g[0][0] * F_g[0] + J_g[1][0] * F_g[1] + J_g[2][0] * F_g[2]
        gradient.q2 = J_g[0][1] * F_g[0] + J_g[1][1] * F_g[1] + J_g[2][1] * F_g[2]
        gradient.q3 = J_g[0][2] * F_g[0] + J_g[1][2] * F_g[1] + J_g[2][2] * F_g[2]
        gradient.q4 = J_g[0][3] * F_g[0] + J_g[1][3] * F_g[1] + J_g[2][3] * F_g[2]
    
        #Normalize the gradient
        gradient.quat_Normalization()

        gradient.quat_scalar(beta)
        q_est_dot = self.quat_sub(q_w,gradient)
        q_est_dot.quat_scalar(delta_t)
        self.q_est = self.quat_add(q_est_prev,q_est_dot)
        self.q_est.quat_Normalization()
        return self.q_est.q1,self.q_est.q2,self.q_est.q3,self.q_est.q4

    '''
        用gyroscope 和 accelerator of gravity to obtain quaternion
    '''
    def obtain_quaternion(self):
        assert len(self.gravity) == len(self.gyroscope)
        self.quaternion = []
        for i in range(len(self.gyroscope)):
            q1,q2,q3,q4=self.imu_filter(self.gravity[i][0],self.gravity[i][1],self.gravity[i][2], \
               self.gyroscope[i][0],self.gyroscope[i][1],self.gyroscope[i][2], )
            self.quaternion.append([q1,q2,q3,q4])
        
        self.quaternion = np.array(self.quaternion)

    '''
        四元数转化为欧拉角
    '''
    def quaternion2euler(self):
        rotation = self.quaternion
        x = rotation[:, 0]
        y = rotation[:, 1]
        z = rotation[:, 2]
        w = rotation[:, 3]
        pitch = np.arcsin(2*(w*y-z*x))
        roll = np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
        yaw = np.arctan2(2*(w*z+x*y),1-2*(z*z+y*y))
        return pitch, roll, yaw
    

    '''
        用accelerometer 和 magnetometer计算欧拉角
    '''
    def obtain_euler(self):
        #TODO:check, accelerometer 用带重力的，以后试试不带重力的
        accelX = self.gravity[:,0]
        accelY = self.gravity[:,1]
        accelZ = self.gravity[:,2]
        magX = self.magnetometer[:,0]
        magY = self.magnetometer[:,1]
        magZ = self.magnetometer[:,2]

        pitch = np.array([atan2(accelY[i], np.sqrt(accelX[i] **2 + accelZ[i]**2)) \
            for i in range(len(accelX))])
        roll = np.array([atan2(-accelX[i], np.sqrt(accelY[i]**2 + accelZ[i]**2)) \
            for i in range(len(accelX))])
        
        Yh = (magY * np.cos(roll)) - (magZ * np.sin(roll))
        Xh = (magX * np.cos(pitch))+(magY * np.sin(roll)*np.sin(pitch)) + (magZ * np.cos(roll) * np.sin(pitch))
        yaw = np.array([atan2(Yh[i],Xh[i]) for i in range(len(magX))])
        return pitch, roll, yaw
    
    def obtain_angle_mlp(self,train_again):
        '''use mlp to predict'''
        if train_again:
            self.mlp_model.train_new()
        p_a = self.mlp_model.test() #角度制度，要变成弧度制
        p_a = np.array([radians(i) for i in p_a])
        return p_a

    '''
        获得手机坐标系与地球坐标系之间的角度（theta）
    '''
    def coordinate_conversion(self):
        gravity = self.gravity
        linear = self.linear

        # g_x = gravity[:, 0]
        g_y = gravity[:, 1]
        g_z = gravity[:, 2]

        # linear_x = linear[:, 0]
        linear_y = linear[:, 1]
        linear_z = linear[:, 2]
        
        theta = np.arctan(np.abs(g_z/g_y))

        # 得到垂直方向加速度（除去g）
        a_vertical = linear_y*np.cos(theta) + linear_z*np.sin(theta)

        return a_vertical
    
    '''
        步数检测函数

        walkType取值：
        normal：正常行走模式
        abnormal：融合定位行走模式（每一步行走间隔大于1s）

        返回值：
        steps
        字典型数组，每个字典保存了峰值位置（index）与该点的合加速度值（acceleration）
    '''
    def step_counter(self, frequency=100, walkType='normal'):
        offset = frequency/100
        g = 9.794
        a_vertical = self.coordinate_conversion()
        slide = 40 * offset # 滑动窗口（100Hz的采样数据）
        frequency = 100 * offset
        # 行人加速度阈值
        min_acceleration = 0.2 * g # 0.2g
        max_acceleration = 2 * g   # 2g
        # 峰值间隔(s)
        # min_interval = 0.4
        min_interval = 0.4 if walkType=='normal' else 3 # 'abnormal
        # max_interval = 1
        # 计算步数
        steps = []
        peak = {'index': 0, 'acceleration': 0}

        # 以40*offset为滑动窗检测峰值
        # 条件1：峰值在0.2g~2g之间
        for i, v in enumerate(a_vertical):
            if v >= peak['acceleration'] and v >= min_acceleration and v <= max_acceleration:
                peak['acceleration'] = v
                peak['index'] = i
            if i%slide == 0 and peak['index'] != 0:
                steps.append(peak)
                peak = {'index': 0, 'acceleration': 0}
        
        # 条件2：两个峰值之前间隔至少大于0.4s*offset
        # del使用的时候，一般采用先记录再删除的原则
        if len(steps)>0:
            lastStep = steps[0]
            dirty_points = []
            for key, step_dict in enumerate(steps):
                # print(step_dict['index'])
                if key == 0:
                    continue
                if step_dict['index']-lastStep['index'] < min_interval*frequency:
                    # print('last:', lastStep['index'], 'this:', step_dict['index'])
                    if step_dict['acceleration'] <= lastStep['acceleration']:
                        dirty_points.append(key)
                    else:
                        lastStep = step_dict
                        dirty_points.append(key-1)
                else:
                    lastStep = step_dict
            
            counter = 0 # 记录删除数量，作为偏差值
            for key in dirty_points:
                del steps[key-counter]
                counter = counter + 1
        
        return steps
    
    # 步长推算
    # 目前的方法不具备科学性，临时使用
    def step_stride(self, max_acceleration):
        return np.power(max_acceleration, 1/4) * 0.5

    # 航向角
    # 根据姿势直接使用yaw,提供初始offset
    def step_heading(self,offset):
        self.obtain_quaternion()
        _, _, yaw = self.quaternion2euler()
        # init_theta = yaw[0] # 初始角度
        for i,v in enumerate(yaw):
            # yaw[i] = -(v-init_theta)
            yaw[i] = -v + offset # 由于yaw逆时针为正向，转化为顺时针为正向更符合常规的思维方式
        return yaw
    
    #从mlp直接学习yaw，其中,yaw是弧度制
    def step_heading_from_mlp(self,offset,train_again = False):
        yaw = self.obtain_angle_mlp(train_again)
        return yaw
    
    '''
        步行轨迹的每一个相对坐标位置, offset表示初始角度
        返回的是预测作为坐标
    '''
    def pdr_position(self, frequency=100, walkType='normal', initPosition=(0, 0),predict_time=0):
        #TODO check! 这里返回弧度制
        offset = radians(self.input[2][predict_time-2])
        self.yaw = self.step_heading_from_mlp(offset,False)
        steps = self.step_counter(frequency=frequency, walkType=walkType)
        position_x = []
        position_y = []
        x = initPosition[0]
        y = initPosition[1]
        position_x.append(x)
        position_y.append(y)
        strides = []
        angle = [offset] #TODO:check!
        time = [0]
        for v in steps:
            index = v['index']
            length = self.step_stride(v['acceleration'])
            strides.append(length)
            theta = self.yaw[index]
            angle.append(degrees(theta))
            x = x + length*np.sin(theta)
            y = y + length*np.cos(theta)
            position_x.append(x)
            position_y.append(y)
            time.append(self.time[index])
        # 步长计入一个状态中，最后一个位置没有下一步，因此步长记为0
        return position_x, position_y, strides + [0], angle,time

    '''
        对于输入data，用10%时间的位置解（相对位置，角）与（经纬度，方向）关系，用来算后面的位置与经纬度关系
    '''
    def predict_position(self,start_index,predic_time:np.ndarray):
        #start_index 表示从time中的哪个index 开始预测，
        #TODO：check!这里我们是去掉表头了！故比写进csv的行号少1！
        noise = 0.2
        x_offset = self.input[:,0][start_index-2]*(1-noise) + self.input[:,0][start_index-3]*noise
        y_offset = self.input[:,1][start_index-2]*(1-noise) + self.input[:,1][start_index-3]*noise
        n_x,n_y,_,angle,step_time=self.pdr_position()    #TODO check!!!突然想起来算的angle是弧度制啊！！

        #interpolate
        tck_x = interpolate.splrep(step_time,np.array(n_x))
        tck_y = interpolate.splrep(step_time,np.array(n_y))

        tmp_time = np.array(predic_time,dtype=np.float64)
        move_x = interpolate.splev(tmp_time,tck_x)
        move_y = interpolate.splev(tmp_time,tck_y)

        ans_x = (np.array(move_x) * 0.000009) + x_offset
        ans_y = (np.array(move_y) * 0.00001141) + y_offset

        #TODO：用时间算出角度
        ans_a = []
        for i in range(len(predic_time)):
            near = [self.yaw[j] for j in range(len(self.time)) if abs(self.time[j]-i)<0.07]
            ans = self.input[2][0] #确保不会报错
            if len(near) != 0:
                ans = fmod(degrees(np.mean(np.array(near)))+720,360)
            ans_a.append(ans)

        #TODO:通过线性拟合的部分
        

        return ans_x,ans_y,ans_a
        


