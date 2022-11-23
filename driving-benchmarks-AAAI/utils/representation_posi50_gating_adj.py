import numpy as np

def acc_multi(acc, speed, pred_speed, multi_max = 3):
    if speed > pred_speed:
        return np.fmax(acc, 0.05)

    diff = pred_speed - speed
    multi = (diff + 6) / 7
    multi = np.fmin(multi, multi_max)

    return np.fmax(acc * multi, 0.05)

def stopping_cnt_set_zero(stopping_cnt, brake, ori_brake, acc, ori_acc):

    if brake > 1 or ori_brake > 0.5 or ori_acc < 0.009:
        stopping_cnt = 0

    return stopping_cnt

def stopping_cnt_adj(stopping_cnt, acc, brake):
    if stopping_cnt > 50:
        acc = 0.61
        brake = 0

    return acc, brake

###################################################################################################
def action_adjusting_carla100(direction, steer, acc, brake, speed, pred_speed, running_cnt, stopping_cnt):
    ''' 기본 셋팅 '''
    ori_acc = acc
    ori_brake = brake
    brake = brake / 1.5
    acc = acc_multi(acc, speed, pred_speed, 3)
    if brake < 0.1:  # or acc > brake:
        brake = 0.0

    if running_cnt < 30:
        steer = 0
        acc *= 0.5

    ''' 위의 조건에 안걸렸으면 '''
    if brake > 0 or ori_brake > 0.5:
        brake = np.fmax(brake, ori_brake)

    stopping_cnt = stopping_cnt_set_zero(stopping_cnt, brake, ori_brake, acc, ori_acc)

    ''' stopping cnt 를 이용한 '''
    acc, brake = stopping_cnt_adj(stopping_cnt, acc, brake)

    ''' 마지막 min, max 셋팅 '''
    acc = np.fmax(np.fmin(acc, 1), 0.0)
    brake = np.fmax(np.fmin(brake * 2, 0.9), 0.0)

    if pred_speed > 30:
        brake = 0

    return steer, acc, brake, stopping_cnt
