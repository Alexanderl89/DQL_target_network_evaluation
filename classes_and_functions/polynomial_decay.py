def polynomial_decay(initial_R,min_R,episode_i,episodes_T,power):
    value = (initial_R-min_R) * (1 - episode_i/episodes_T)**power + min_R
    return value