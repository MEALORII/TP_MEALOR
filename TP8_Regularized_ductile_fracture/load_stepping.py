import numpy as np


class LoadSteppingStrategy:
    def __init__(
        self,
        target_df,
        f_ini,
        dU_min,
        dU_max,
        max_load_increase_factor=2,
        max_load_decrease_factor=4,
        target_window_factor=1.2,
    ):
        self.target_df = target_df
        self.dU_min = dU_min
        self.dU_max = dU_max
        self.f_max_old = f_ini
        self.max_load_increase_factor = max_load_increase_factor
        self.max_load_decrease_factor = max_load_decrease_factor
        self.target_window_factor = target_window_factor

    def new_step(self, dU, f_max):
        self.df_max = abs(f_max - self.f_max_old)
        self.f_max_old = np.copy(f_max)
        load_decrease_factor = min(
            1 + self.df_max / (self.target_df / 10), self.max_load_decrease_factor
        )
        load_increase_factor = max(
            self.max_load_increase_factor - self.df_max / self.target_df, 1
        )
        if self.df_max > self.target_window_factor * self.target_df:
            # print("Decrease factor", load_decrease_factor)
            # print(dU, dU * load_decrease_factor)
            dU /= load_decrease_factor
        elif self.df_max < self.target_df / self.target_window_factor:
            # print("Increase factor", load_increase_factor)
            # print(dU)
            dU *= load_increase_factor
            # print("Increase factor", load_increase_factor)
        # else:
        #     print("Do nothing")
        return max(min(dU, self.dU_max), self.dU_min)
