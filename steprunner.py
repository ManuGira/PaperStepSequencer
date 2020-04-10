import numpy as np
import midiplayer
import cv2 as cv
import time

class StepRunner:
    def __init__(self, pixpermm):
        self.pixpermm = pixpermm
        # top left corner of the grid
        self.grid_pos_xy = np.array([16, 18]) * self.pixpermm
        self.grid_size_xy = np.array([48, 24])

        self.nb_steps = np.array([16, 14, 8, 6])
        self.nb_rows = len(self.nb_steps)

        # size in warped pixel of a grid square
        self.grid_square_size_x = np.int32(self.grid_size_xy[0] / self.nb_steps * self.pixpermm)
        self.grid_square_size_y = np.int32(self.grid_size_xy[1] / self.nb_rows * self.pixpermm)

        self.bpm = 120
        self.beats_per_grid = 4
        self.full_grid_period = (60/self.bpm)*self.beats_per_grid  # should be 2s for 16 steps at 120bmp

        # the current step of each row
        self.rows_current_step = np.array([-1] * self.nb_rows)
        self.midiplayer = midiplayer.MidiPlayer()

        self.entries = []
        self.entries_grid = [[0]*nb_step for nb_step in self.nb_steps]
        self.entries_max_hit = 9

    def draw_current_step(self, frame_warped):

        frame_tmp = frame_warped.copy()

        # lets highlight a column:
        for k in range(self.nb_rows):
            # recover top left corner of rectangle
            x = np.int32(self.grid_pos_xy[0] + self.rows_current_step[k] * self.grid_square_size_x[k])
            # print("Camera self.sequencer_prev_step", self.sequencer_prev_step)
            y = np.int32(self.grid_pos_xy[1] + self.grid_square_size_y*k)
            dx = self.grid_square_size_x[k]
            dy = self.grid_square_size_y
            rectangle_corners = np.array([
                [     x,      y],
                [x + dx,      y],
                [x + dx, y + dy],
                [     x, y + dy]
            ])

            cv.polylines(frame_tmp, [rectangle_corners], True, (0, 0, 255), thickness=3)
            frame_warped = np.uint8(0.5 * np.float32(frame_warped) + 0.5 * np.float32(frame_tmp))
        return frame_warped

    def run(self):
        # grid_length = self.grid_dim_xy[0]
        #
        # # unit in seconds
        # # 1 mesure
        # beat_period = 60 / self.bpm  # seconds (0.5)
        # # 1 step
        # one_step_period = beat_period / 4  # (0.125)
        # full_grid_period = one_step_period * 16  # (2.0)

        # list of rdv times. Its a dictionnary rdv_time->row
        agenda = {}
        for row, nb_step in enumerate(self.nb_steps):
            times = self.full_grid_period*np.arange(nb_step)/nb_step
            for step, rdv_time in enumerate(times):
                # for each rdv time, we want to know which rows is involved, and which step it is for this row
                if rdv_time not in agenda.keys():
                    agenda[rdv_time] = []
                rdv = {
                    "row_id": row,
                    "step": step
                }
                agenda[rdv_time].append(rdv)
        # extract the list of rdv times.
        rdvs_times = sorted(list(agenda.keys()))

        ts0 = time.time()
        tss = []

        current_rdv_id = -1
        while True:
            current_rdv_id = (current_rdv_id+1)%len(rdvs_times)

            # which rows must be triggered now?
            rdv_time = rdvs_times[current_rdv_id]
            for rdv in agenda[rdv_time]:
                # increase row current steps and trigger its midi message
                row, step = rdv["row_id"], rdv["step"]
                self.rows_current_step[row] = (self.rows_current_step[row] + 1) % self.nb_steps[row]
                if self.entries_grid[row][step] >= 7:  # TODO: don't use this comparison
                    self.midiplayer.note_on(row)

            # must plan our next rdv
            next_rdv_id = (current_rdv_id+1) % len(rdvs_times)
            next_rdv_time = rdvs_times[next_rdv_id]
            # sleep until next rdv
            ts = (time.time()-ts0)
            sleep_duration = (next_rdv_time-ts)%self.full_grid_period
            time.sleep(sleep_duration)

        # while True:
        #     ts = time.time()-ts0
        #     step_ts = (ts % self.full_grid_period)
        #
        #     step = int(step_ts)
        #     if step == self.sequencer_prev_step:
        #         step += 1
        #     if step % grid_length != (self.sequencer_prev_step + 1) % grid_length:
        #         print("ERROR")
        #     to_wait = (step + 1 - step_ts) * one_step_period
        #
        #     # cv.waitKey(int(to_wait * 1000))
        #     time.sleep(to_wait)
        #     self.sequencer_prev_step = step
        #     self.midiplayer.note_off_all()
        #
        #     for entry in self.entries:
        #         entry_step, percu_id = entry
        #         if entry_step == step:
        #             self.midiplayer.note_on(percu_id)
        #
        #     ts2 = time.time()
        #     tss.append(ts2)
        #     # print("Sequencer self.sequencer_prev_step", self.sequencer_prev_step)
        #     # print("step:", self.sequencer_prev_step, ", to wait: ", to_wait)