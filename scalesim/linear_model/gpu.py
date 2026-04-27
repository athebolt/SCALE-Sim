def ga10b_linear_model(total_cycles, s_row=None, s_col=None, t_time=None):
    """
    Time linear model for Ampere GA10b (Jetson Orin Nano).
    The hardware frequency is inherently tied to the chip architecture.
    """
    
    GA10B_FREQ_MHZ = 1300 
    
    # Time (us) = Cycles / Frequency (MHz)
    time_us = total_cycles / GA10B_FREQ_MHZ
    
    return time_us