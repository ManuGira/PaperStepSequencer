import time
import matplotlib.pyplot as plt
# import cv2 as cv
import numpy as np


def squence(N, bpm, steps_per_beats):
    grid_length = 16

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
        print("step:", step, ", to wait: ", to_wait, flush=True)
    return tss

def compute_error(tss, bpm, steps_per_beats):
    # periods in seconds
    beat_period = 60 / bpm
    one_step_period = beat_period / steps_per_beats

    # one_step_perdiod is the slope of tss
    tss = np.array(tss)
    ideal_tss = np.array(list(range(len(tss)))) * one_step_period

    # center ideal_tss on tss minimizing the square error
    ideal_tss += np.mean((tss-ideal_tss)**2)**0.5

    rmse = np.mean((tss-ideal_tss)**2)**0.5
    print(f"Error: {rmse:.6f} seconds -> {100*rmse/one_step_period:.2f}% of a step")

    return tss-ideal_tss

def main():
    bpm = 120
    steps_per_beats = 4
    tss = squence(32*10, bpm, steps_per_beats)
    error_tss = compute_error(tss, bpm, steps_per_beats)
    plt.plot(error_tss)
    plt.show()

if __name__ == '__main__':
    main()

    # time.sleep @ 320: (120, 4) -> Error: 0.000417 seconds -> 0.33% of a step
    # cv.waitKey @ 320: (120, 4) -> Error: 0.000585 seconds -> 0.47% of a step
