import numpy as np
import midiplayer
import cv2 as cv
import time

class StepRunner:
    def __init__(self, pixpermm):
        self.pixpermm = pixpermm
        # top left corner of the grid
        self.grid_pos_xy = np.array([16, 18]) * self.pixpermm
        # size in warped pixel of a grid square
        self.grid_square_size_xy = np.array([3, 6]) * self.pixpermm
        # dimensionality of the grid
        self.grid_dim_xy = np.array([16, 4])

        self.bpm = 120
        self.steps_per_beats = 4
        self.sequencer_prev_step = -1
        self.midiplayer = midiplayer.MidiPlayer()

        self.entries = []
        self.entries_grid = np.zeros(self.grid_dim_xy).transpose()
        self.entries_max_hit = 9

    def draw_current_step(self, frame_warped):

        frame_tmp = frame_warped.copy()

        # lets highlight a column:
        # recover top left corner of column
        x = self.grid_pos_xy[0] + self.sequencer_prev_step * self.grid_square_size_xy[0]
        # print("Camera self.sequencer_prev_step", self.sequencer_prev_step)
        y = self.grid_pos_xy[1]
        dx = self.grid_square_size_xy[0]
        dy = self.grid_square_size_xy[1] * self.grid_dim_xy[1]
        rectangle_corners = np.array([
            [x, y],
            [x + dx, y],
            [x + dx, y + dy],
            [x, y + dy]
        ])

        cv.polylines(frame_tmp, [rectangle_corners], True, (0, 0, 255), thickness=3)
        frame_warped = np.uint8(0.5 * np.float32(frame_warped) + 0.5 * np.float32(frame_tmp))
        return frame_warped

    def run(self):
        grid_length = self.grid_dim_xy[0]

        # unit in seconds
        beat_period = 60 / self.bpm  # seconds (0.5)
        one_step_period = beat_period / self.steps_per_beats  # (0.125)
        full_grid_period = one_step_period * grid_length  # (2.0)

        tss = []
        while True:
            ts = time.time()
            step_ts = (ts % full_grid_period) / one_step_period
            step = int(step_ts)
            if step == self.sequencer_prev_step:
                step += 1
            if step % grid_length != (self.sequencer_prev_step + 1) % grid_length:
                print("ERROR")
            to_wait = (step + 1 - step_ts) * one_step_period

            # cv.waitKey(int(to_wait * 1000))
            time.sleep(to_wait)
            self.sequencer_prev_step = step
            self.midiplayer.note_off_all()

            for entry in self.entries:
                entry_step, percu_id = entry
                if entry_step == step:
                    self.midiplayer.note_on(percu_id)

            ts2 = time.time()
            tss.append(ts2)
            # print("Sequencer self.sequencer_prev_step", self.sequencer_prev_step)
            # print("step:", self.sequencer_prev_step, ", to wait: ", to_wait)