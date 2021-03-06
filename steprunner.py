import numpy as np
import midiplayer
import cv2 as cv
import time

class StepRunner:
    def __init__(self, pixpermm, frame_shape):
        self.pixpermm = pixpermm
        # top left corner of the grid
        self.grid_pos_xy = np.array([16, 18]) * self.pixpermm
        self.grid_size_xy = np.array([48, 24]) * self.pixpermm

        self.nb_steps = np.array([4, 4, 6, 6])
        self.nb_rows = len(self.nb_steps)

        # size in warped pixel of a grid square
        self.grid_square_size_x = self.grid_size_xy[0] / self.nb_steps
        self.grid_square_size_y = self.grid_size_xy[1] / self.nb_rows

        self.ts0 = 0
        self.bpm = 240
        self.beats_per_grid = 4
        self.full_grid_period = (60/self.bpm)*self.beats_per_grid  # should be 2s for 16 steps at 120bmp

        # the current step of each row
        self.rows_current_step = np.array([-1] * self.nb_rows)
        self.midiplayer = midiplayer.MidiPlayer()

        self.entries = []
        self.entries_grid = [[0]*nb_step for nb_step in self.nb_steps]
        self.entries_max_hit = 9

        self.ar_content = np.zeros(shape=frame_shape[::-1], dtype=np.uint8)
        self.ar_content_grid = np.zeros(shape=frame_shape[::-1], dtype=np.uint8)
        self.ar_content_constant = np.zeros(shape=frame_shape[::-1], dtype=np.uint8)
        self.is_update_required = True
        self.init_ar_content_constant()

    def get_entries_rectangles(self, entries):
        rectangles = []
        for row_step in entries:
            row, step = row_step
            dx = self.grid_square_size_x[row]
            dy = self.grid_square_size_y
            y = self.grid_pos_xy[1] + row * dy
            x = self.grid_pos_xy[0] + step * dx
            rect = np.array([
                [x, y],
                [x + dx, y],
                [x + dx, y + dy],
                [x, y + dy]], dtype=np.int32)
            rectangles.append(rect)
        return rectangles

    def get_entries_halfcircle(self, entries):
        halfcircles = []
        for row_step in entries:
            row, step = row_step
            dx = self.grid_square_size_x[row]
            dy = self.grid_square_size_y
            dy2 = dy/2
            dy3 = dy/3
            y = self.grid_pos_xy[1] + row * dy
            x = self.grid_pos_xy[0] + step * dx
            center = int(x), int(y+dy2)
            axes = int(dy3), int(dy3)  # [[0, dy3], [dy3, 0]]
            angle = 0
            arcStart = -90
            arcEnd = 90
            delta = 10
            pts = cv.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta)
            halfcircles.append(pts)
        return halfcircles

    def init_ar_content_constant(self):
        all_entries = []
        for row, nb_steps in enumerate(self.nb_steps):
            for step in range(nb_steps):
                entry = [row, step]
                all_entries.append(entry)
        pts_list = self.get_entries_rectangles(all_entries)
        pts_list += self.get_entries_rectangles(all_entries)
        cv.polylines(self.ar_content_constant, pts_list, True, (255,), thickness=1)

    def update_ar_content(self):
        # Highlight current step
        if self.is_update_required:
            self.ar_content_grid = self.ar_content_constant.copy()
            self.is_update_required = False
            pts_list = self.get_entries_halfcircle([[k, self.rows_current_step[k]] for k in range(self.nb_rows)])
            cv.polylines(self.ar_content_grid, pts_list, True, (255,), thickness=3)

        # timeline
        self.ar_content = self.ar_content_grid.copy()
        ts = ((time.time()-self.ts0)/self.full_grid_period) % 1
        x = int(self.grid_pos_xy[0] + ts*self.grid_size_xy[0])
        y0 = int(self.grid_pos_xy[1] - self.grid_square_size_y/2)
        y1 = int(self.grid_pos_xy[1] + self.grid_size_xy[1] + self.grid_square_size_y/2)
        cv.line(self.ar_content, (x, y0), (x, y1), color=(255,))

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
            times = np.round(times, decimals=10)  # rough round to avoid rounding errors
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

        self.ts0 = time.time()

        current_rdv_id = -1
        while True:
            current_rdv_id = (current_rdv_id+1)%len(rdvs_times)

            # which rows must be triggered now?
            rdv_time = rdvs_times[current_rdv_id]
            if len(agenda[rdv_time])>0:
                self.is_update_required = True
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
            ts = (time.time()-self.ts0)
            sleep_duration = (next_rdv_time-ts)%self.full_grid_period

            time.sleep(sleep_duration)

