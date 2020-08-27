import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, default=None, help='timing e.g. 21:30')
opts = parser.parse_args()

if opts.time is None:
    wait_time = 0
else:
    target_time = opts.time.split(':')
    assert 0 <= int(target_time[0]) <= 24, 'Hour is error'
    assert 0 <= int(target_time[1]) <= 59, 'Minute is error'
    cur_time = time.strftime('%H:%M:%S', time.localtime()).split(':')
    wait_time = ((int(target_time[0]) - int(cur_time[0]) + 24) % 24) * 3600
    wait_time += (int(target_time[1]) - int(cur_time[1])) * 60
    wait_time += 0 - int(cur_time[2])
    if (wait_time < 0):
        wait_time += 24 * 3600

print('Sleep until', opts.time)
time.sleep(wait_time)

# works after timing
os.system("python /home/wyq/exp/my_research/main_baseline.py")
