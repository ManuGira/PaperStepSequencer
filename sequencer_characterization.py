import time
import matplotlib.pyplot as plt
import cv2 as cv


def squence(N):
    bpm = 120
    steps_per_beats = 4
    grid_length = 16

    beats_per_grid = grid_length/steps_per_beats # (4)
    beat_period = 60/bpm # seconds (0.5)
    one_step_period = beat_period/steps_per_beats  # (0.125)
    full_grid_period = one_step_period*grid_length  # (2.0)

    tss = []
    prev_step = -1
    for k in range(N):
        ts = time.time()
        step_ts = (ts%full_grid_period)/one_step_period
        step = int(step_ts)
        if step==prev_step:
            step += 1
        if step%grid_length != (prev_step+1)%grid_length:
            print("ERROR")
        prev_step = step
        to_wait = (step+1-step_ts)*one_step_period
        # cv.waitKey(int(to_wait * 1000))
        time.sleep(to_wait)
        ts2 = time.time()
        tss.append(ts2)
        print("step:", step, ", to wait: ", to_wait)
    return tss

def compute_error():
    pass

def main():
    tss = squence(32)
    plt.plot(tss)
    plt.show()

if __name__ == '__main__':
    main()