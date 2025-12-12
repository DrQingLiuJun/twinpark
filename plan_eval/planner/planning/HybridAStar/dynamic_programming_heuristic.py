"""

A* grid based planning

author: Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import heapq
import math

import matplotlib.pyplot as plt
from HybridAStar.car import check_car_collision

show_animation = False


class Node:

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(
            self.cost) + "," + str(self.parent_index)


def calc_final_path(goal_node, closed_node_set, resolution):
    # generate final course
    rx, ry = [goal_node.x * resolution], [goal_node.y * resolution]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_node_set[parent_index]
        rx.append(n.x * resolution)
        ry.append(n.y * resolution)
        parent_index = n.parent_index

    return rx, ry


def calc_distance_heuristic(gx, gy, ox, oy, resolution, rr, kd_tree):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    resolution: grid resolution [m]
    rr: robot radius[m]
    """
    
    goal_node = Node(round(gx / resolution), round(gy / resolution), 0.0, -1)

    ox = [iox / resolution for iox in ox]
    oy = [ioy / resolution for ioy in oy]

    obstacle_map, min_x, min_y, max_x, max_y, x_w, y_w = calc_obstacle_map(
        ox, oy, resolution, rr, kd_tree)

    motion = get_motion_model()

    open_set, closed_set = dict(), dict()
    open_set[calc_index(goal_node, x_w, min_x, min_y)] = goal_node
    priority_queue = [(0, calc_index(goal_node, x_w, min_x, min_y))]

    while True:
        if not priority_queue:
            break
        cost, c_id = heapq.heappop(priority_queue)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        # show graph
        if show_animation:  # pragma: no cover
            #! modifued by zzs 2024/3/14
            # plt.plot( current.x * resolution, current.y * resolution, "xc")
            # plt.plot( (current.x+min(ox)) * resolution, (current.y+min(oy)) * resolution, "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closed_set.keys()) % 10 == 0:
                plt.pause(0.001)
        # Remove the item from the open set

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            node = Node(current.x + motion[i][0],
                        current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)
            n_id = calc_index(node, x_w, min_x, min_y)

            if n_id in closed_set:
                continue

            if not verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
                continue

            if n_id not in open_set:
                open_set[n_id] = node  # Discover a new node
                heapq.heappush(
                    priority_queue,
                    (node.cost, calc_index(node, x_w, min_x, min_y)))
            else:
                if open_set[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    open_set[n_id] = node
                    heapq.heappush(
                        priority_queue,
                        (node.cost, calc_index(node, x_w, min_x, min_y)))

    return closed_set


def verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
    if node.x < min_x:
        return False
    elif node.y < min_y:
        return False
    elif node.x >= max_x:
        return False
    elif node.y >= max_y:
        return False

    #! modified by zzs 2024/3/15
    if obstacle_map[node.x-min_x][node.y-min_y]:
    # original version
    #if obstacle_map[node.x][node.y]:   
        return False

    return True


def calc_obstacle_map(ox, oy, resolution, vr, kd_tree):
    # min_x = round(min(ox))
    # min_y = round(min(oy))
    # max_x = round(max(ox))
    # max_y = round(max(oy))

    # x_width = round(max_x - min_x)
    # y_width = round(max_y - min_y)
    # modified by lm
    min_x = int(math.floor(min(ox)))
    min_y = int(math.floor(min(oy)))
    max_x = int(math.ceil(max(ox)))
    max_y = int(math.ceil(max(oy)))

    # 网格宽高用“包含端点”的写法，避免边界漏格
    x_width = int(max_x - min_x + 1)
    y_width = int(max_y - min_y + 1)

    # obstacle map generation
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
    for ix in range(x_width):
        x = ix + min_x
        for iy in range(y_width):
            y = iy + min_y
            #  print(x, y)
            # ! modified by zzs 2023/12/12
            x_list, y_list, yaw_list = [], [], []
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(0.0)
            #! modifued by zzs 2024/3/15
            # check_collision = check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree)
            # if check_collision == True:
            #     obstacle_map[ix][iy] = True

            #original version
            if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
                obstacle_map[ix][iy] = True
            # for iox, ioy in zip(ox, oy):
            #     d = math.hypot(iox - x, ioy - y)
            #     # original version
            #     # if d <= vr / resolution:
            #     if d <= 0.2:
            #         obstacle_map[ix][iy] = True
            #         break

    return obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width


def calc_index(node, x_width, x_min, y_min):
    return (node.y - y_min) * x_width + (node.x - x_min)


def get_motion_model():
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion
