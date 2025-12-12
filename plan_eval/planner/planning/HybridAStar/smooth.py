import casadi as ca
import numpy as np
import math
from scipy.spatial import cKDTree
from HybridAStar.car import BUBBLE_R
from HybridAStar.car import BUBBLE_DIST
from HybridAStar.car import rectangle_check

w1 = 2 #smooth
w2 = 1 #length
w3 = 10 #yaw
w4 = 0.5 #deviation

def calculate_angle_ca(x1, y1, x2, y2):
    # 计算水平和垂直差异
    dx = x2 - x1
    dy = y2 - y1
    
    # 使用arctan函数计算弧度
    radian = ca.atan2(dy, dx)

    return radian

def calculate_angle(x1, y1, x2, y2):
    # 计算水平和垂直差异
    dx = x2 - x1
    dy = y2 - y1
    
    # 使用arctan函数计算弧度
    radian = math.atan2(dy, dx)

    return radian

def path_optimization(Path, ox, oy):

    #前进段与后退段分别优化
    index_dir_change = [] #每一段最后一个点的索引
    for i in range(1, len(Path.direction_list)):
        if Path.direction_list[i] != Path.direction_list[i-1]:
            index_dir_change.append(i-1)
    index_dir_change.append(len(Path.direction_list)-1)

    
    x_list_opti, y_list_opti, yaw_list_opti = [], [], [Path.yaw_list][0]
    for i in range(len(index_dir_change)):
        if i == 0:
            start = 0
            end = index_dir_change[i]
            opti = ca.Opti()
            # 创建优化变量
            x = opti.variable(len(Path.x_list[start:end])+1) 
            y = opti.variable(len(Path.y_list[start:end])+1)
            # yaw = opti.variable(len(Path.yaw_list[start:end])+1)      # yaw 不是优化变量，不用opti.variable
            yaw = list(range(len(Path.x_list[start:end])+1))
            opti.set_initial(x, Path.x_list[start:end+1])
            opti.set_initial(y, Path.y_list[start:end+1])
            # opti.set_initial(yaw, Path.yaw_list[start:end+1])

            # 设置约束
            # 1.首尾固定，不优化
            opti.subject_to(x[0] == Path.x_list[start])
            opti.subject_to(x[-1] == Path.x_list[end])
            opti.subject_to(y[0] == Path.y_list[start])
            opti.subject_to(y[-1] == Path.y_list[end])
            # opti.subject_to(yaw[0] == Path.yaw_list[start])
            # opti.subject_to(yaw[-1] == Path.yaw_list[end])

            # 2.与原路径偏离程度,避障安全范围
            for i in range(1,end):
                opti.subject_to(x[i] >= Path.x_list[start+i] - 0.2)
                opti.subject_to(x[i] <= Path.x_list[start+i] + 0.2)
                opti.subject_to(y[i] >= Path.y_list[start+i] - 0.2)
                opti.subject_to(y[i] <= Path.y_list[start+i] + 0.2)
                # for i in range(1, end):
                #     if Path.direction_list[i] == True:
                #         yaw[i] = calculate_angle_ca(x[i-1],y[i-1],x[i],y[i])
                #     if Path.direction_list[i] == False:
                #         yaw[i] = calculate_angle_ca(x[i],y[i],x[i-1],y[i-1])
                # yaw[i] == calculate_angle_ca(x[i-1],y[i-1],x[i],y[i])

            # 3.曲率约束
            for i in range(2,len(Path.x_list[start:end])):
                opti.subject_to(((x[i-2]-2*x[i-1]+x[i])**2 + (y[i-2]-2*y[i-1]+y[i])**2) <= 0.25*0.25*0.3)

            # 设置代价函数
            cost_total = 0
            # 1.平滑度代价
            cost_smooth = 0
            for i in range(1,end):
                cost_smooth += (x[i-1]-2*x[i]+x[i+1])**2 + (y[i-1]-2*y[i]+y[i+1])**2

            # 2.路径长度代价
            cost_length = 0
            for i in range(1,end+1):
                cost_length += (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2

            # 3.航向角代价
            cost_yaw = 0
            # for i in range(1,end):
            #     yaw[i] = calculate_angle(x[i-1],y[i-1],x[i],y[i])
            #     cost_yaw += (yaw[i]-yaw[i-1])**2
            if Path.direction_list[start+1] == True:
                yaw[0] = calculate_angle_ca(x[0],y[0],x[1],y[1])
                yaw[-1] = calculate_angle_ca(x[-2],y[-2],x[-1],y[-1])
            else:
                yaw[0] = calculate_angle_ca(x[1],y[1],x[0],y[0])
                yaw[-1] = calculate_angle_ca(x[-1],y[-1],x[-2],y[-2])
            
            cost_yaw += (yaw[0]-Path.yaw_list[1])**2 + (yaw[-1]-Path.yaw_list[end])**2

            # 4.偏离代价
            cost_deviation = 0
            # for i in range(1,end):
            #     cost_deviation += (x[i]-Path.x_list[i])**2 + (y[i]-Path.y_list[i-1])**2

            # 求解优化问题
            cost_total =  w1 * cost_smooth + w2 * cost_length + w3 * cost_yaw + w4 * cost_deviation
            opti.minimize(cost_total)
            opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':1}
            opti.solver('ipopt', opts_setting)
            sol = opti.solve()
            x_list_opti.extend(list(sol.value(x)))
            y_list_opti.extend(list(sol.value(y)))
            # yaw_list_opti.extend(list(sol.value(yaw)))
            # for i in range(1, len(Path.x_list[start:end])+1):
            #     if Path.direction_list[i] == True:
            #         yaw_list_opti.append(calculate_angle(x_list_opti[i-1],y_list_opti[i-1],x_list_opti[i],y_list_opti[i]))
            #     if Path.direction_list[i] == False:
            #         yaw_list_opti.append(calculate_angle(x_list_opti[i],y_list_opti[i],x_list_opti[i-1],y_list_opti[i-1]))

        else:
            start = index_dir_change[i-1]
            end = index_dir_change[i]
            opti = ca.Opti()
            # 创建优化变量
            x = opti.variable(len(Path.x_list[start:end])+1) 
            y = opti.variable(len(Path.y_list[start:end])+1)
            # yaw = opti.variable(len(Path.yaw_list[start:end])+1)
            yaw = list(range(len(Path.x_list[start:end])+1))
            if end == len(Path.x_list)-1:
                opti.set_initial(x, Path.x_list[start:])
                opti.set_initial(y, Path.y_list[start:])
            else:
                opti.set_initial(x, Path.x_list[start:end+1])
                opti.set_initial(y, Path.y_list[start:end+1])
            # opti.set_initial(yaw, Path.yaw_list[start:end+1])

            # 设置约束
            # 1.首尾固定，不优化
            opti.subject_to(x[0] == Path.x_list[start])
            opti.subject_to(x[-1] == Path.x_list[end])
            opti.subject_to(y[0] == Path.y_list[start])
            opti.subject_to(y[-1] == Path.y_list[end])
            # opti.subject_to(yaw[0] == Path.yaw_list[start])
            # opti.subject_to(yaw[-1] == Path.yaw_list[end])

            # 2.与原路径偏离程度，避障安全范围
            for i in range(1,len(Path.x_list[start:end])):
                opti.subject_to(x[i] >= Path.x_list[start+i] - 0.2)
                opti.subject_to(x[i] <= Path.x_list[start+i] + 0.2)
                opti.subject_to(y[i] >= Path.y_list[start+i] - 0.2)
                opti.subject_to(y[i] <= Path.y_list[start+i] + 0.2)
                # for i in range(1, len(Path.x_list[start:end])):
                #     if Path.direction_list[i] == True:
                #         yaw[i] = calculate_angle_ca(x[i-1],y[i-1],x[i],y[i])
                #     if Path.direction_list[i] == False:
                #         yaw[i] = calculate_angle_ca(x[i],y[i],x[i-1],y[i-1]) 
                # yaw[i] == calculate_angle_ca(x[i-1],y[i-1],x[i],y[i])

            # 3.航向角
                # opti.subject_to(yaw[i] == calculate_angle(x[i],y[i],x[i-1],y[i-1]))


            # 4.曲率约束
            for i in range(2,len(Path.x_list[start:end])):
                opti.subject_to(((x[i-2]-2*x[i-1]+x[i])**2 + (y[i-2]-2*y[i-1]+y[i])**2) <= 0.25*0.25*0.3)

            # 设置代价函数
            cost_total = 0
            # 1.平滑度代价
            cost_smooth = 0
            for i in range(1,len(Path.x_list[start:end])):
                cost_smooth += (x[i-1]-2*x[i]+x[i+1])**2 + (y[i-1]-2*y[i]+y[i+1])**2

            # 2.路径长度代价
            cost_length = 0
            # for i in range(1,len(Path.x_list[start:end])+1): #+1因为要考虑到最后一个点
            #     cost_length += (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2

            # 3.航向角代价
            cost_yaw = 0
            # for i in range(1,len(Path.x_list[start:end])+1):
            #     yaw[i] = calculate_angle(x[i-1],y[i-1],x[i],y[i])
            #     cost_yaw += (yaw[i]-yaw[i-1])**2
            if Path.direction_list[start+1] == True:
                yaw[0] = calculate_angle_ca(x[0],y[0],x[1],y[1])
                yaw[-1] = calculate_angle_ca(x[-2],y[-2],x[-1],y[-1])
                # yaw[1] = calculate_angle_ca(x[1],y[1],x[2],y[2])
                # yaw[-2] = calculate_angle_ca(x[-3],y[-3],x[-2],y[-2])
            else:
                yaw[0] = calculate_angle_ca(x[1],y[1],x[0],y[0])
                yaw[-1] = calculate_angle_ca(x[-1],y[-1],x[-2],y[-2])
                # yaw[1] = calculate_angle_ca(x[2],y[2],x[1],y[1])
                # yaw[-2] = calculate_angle_ca(x[-2],y[-2],x[-3],y[-3])
            
            if end == len(Path.x_list)-1:
                cost_yaw += (yaw[-1]-Path.yaw_list[end])**2
                # pass
            else:
                cost_yaw += (yaw[0]-Path.yaw_list[start])**2 + (yaw[-1]-Path.yaw_list[end])**2

            # 4.偏离代价
            cost_deviation = 0
            # for i in range(1,len(Path.x_list[start:end])):
            #     cost_deviation += (x[i]-Path.x_list[start+i])**2 + (y[i]-Path.y_list[start+i-1])**2
        
            # 求解优化问题
            cost_total =  w1 * cost_smooth + w3 * cost_yaw + w4 * cost_deviation + w2 * cost_length
            opti.minimize(cost_total)
            opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':1}
            opti.solver('ipopt', opts_setting)
            # opti.solver('ipopt')
            sol = opti.solve()
            del x_list_opti[-1] #去掉重复的端点
            del y_list_opti[-1]
            # del yaw_list_opti[-1]
            x_list_opti.extend(list(sol.value(x)))
            y_list_opti.extend(list(sol.value(y)))
            # yaw_list_opti.extend(list(sol.value(yaw)))
            # for i in range(1, len(Path.x_list[start:end])+1):
            #     if Path.direction_list[i] == True:
            #         yaw_list_opti.append(calculate_angle(x_list_opti[i-1],y_list_opti[i-1],x_list_opti[i],y_list_opti[i]))
            #     if Path.direction_list[i] == False:
            #         yaw_list_opti.append(calculate_angle(x_list_opti[i],y_list_opti[i],x_list_opti[i-1],y_list_opti[i-1]))

    # opti = ca.Opti()
    # # 创建优化变量
    # x = opti.variable(len(Path.x_list),1) 
    # y = opti.variable(len(Path.y_list),1)
    # yaw = opti.variable(len(Path.yaw_list),1)
    # opti.set_initial(x, Path.x_list)
    # opti.set_initial(y, Path.y_list)
    # opti.set_initial(yaw, Path.yaw_list)

    # # 设置约束
    # # 1.首尾固定，不优化
    # opti.subject_to(x[0] == Path.x_list[0])
    # opti.subject_to(x[-1] == Path.x_list[-1])
    # opti.subject_to(y[0] == Path.y_list[0])
    # opti.subject_to(y[-1] == Path.y_list[-1])
    # opti.subject_to(yaw[0] == Path.yaw_list[0])
    # opti.subject_to(yaw[-1] == Path.yaw_list[-1])

    # # 2.与原路径偏离程度
    # for i in range(1,len(Path.x_list)-1):
    #     opti.subject_to(x[i] >= Path.x_list[i] - 0.2)
    #     opti.subject_to(x[i] <= Path.x_list[i] + 0.2)
    #     opti.subject_to(y[i] >= Path.y_list[i] - 0.2)
    #     opti.subject_to(y[i] <= Path.y_list[i] + 0.2)

    # # 3.曲率约束
    # # for i in range(1,len(Path.x_list)-1):
    # #     opti.subject_to(((x[i-1]-2*x[i]+x[i+1])**2 + (y[i-1]-2*y[i]+y[i+1])**2) <= 0.3*0.4)

    # # 设置代价函数
    # cost_total = 0
    # # 1.平滑度代价
    # cost_smooth = 0,
    # for i in range(1,len(Path.x_list)-1):
    #     cost_smooth += (x[i-1]-2*x[i]+x[i+1])**2 + (y[i-1]-2*y[i]+y[i+1])**2

    # # 2.路径长度代价
    # cost_length = 0
    # for i in range(1,len(Path.x_list)):
    #     cost_length += (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2

    # # 3.航向角代价
    # cost_yaw = 0
    # # for i in range(1,len(Path.x_list)):
    # #     yaw[i] = calculate_angle(x[i],y[i],x[i-1],y[i-1])
    # #     cost_yaw += (yaw[i]-yaw[i-1])**2

    # # 求解优化问题
    # cost_total =  w1 * cost_smooth + w2 * cost_length + w3 * cost_yaw
    # opti.minimize(cost_total)
    # opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':1}
    # opti.solver('ipopt', opts_setting)
    # sol = opti.solve()
    # Path.x_list = list(sol.value(x))
    # Path.y_list = list(sol.value(y))
                    
    #yaw没有参与优化， yaw_list_opti.extend(list(sol.value(yaw)))， yaw_list_opti 和 Path.yaw_list 是一样的
            
    Path.x_list = x_list_opti
    Path.y_list = y_list_opti
    # Path.yaw_list = yaw_list_opti     
    for i in range(1, len(Path.x_list)-1):
        if Path.direction_list[i] == True:
            Path.yaw_list[i] = calculate_angle(Path.x_list[i-1],Path.y_list[i-1],Path.x_list[i],Path.y_list[i])
        if Path.direction_list[i] == False:
            Path.yaw_list[i] = calculate_angle(Path.x_list[i],Path.y_list[i],Path.x_list[i-1],Path.y_list[i-1])

    return Path
