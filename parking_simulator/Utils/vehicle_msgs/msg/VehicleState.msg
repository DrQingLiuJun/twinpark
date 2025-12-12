# 车辆状态（后轴中心坐标系）
float64 x          # 位置 X (m)
float64 y          # 位置 Y (m)
float64 yaw        # 航向角 (rad)
float64 vx         # 纵向速度 (m/s)
float64 vy         # 横向速度 (m/s)
float64 omega      # 角速度 (rad/s)
int32 gear         # 档位 (1=前进, -1=倒车, 0=空挡)
bool reverse       # 倒车标志
time stamp         # 时间戳
