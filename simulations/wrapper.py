import logging

logger = logging.getLogger(__name__)

import numpy as np

from simulations.simulate_ps import SimulateMap


def simulator(theta, dnds_list, s_ary, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, mask_roi, psf_r_func, exp_map):
    the_map = np.zeros(np.sum(~mask_sim))
    aux_vars = np.zeros(2)

    good_map = False  # Check so map doesn't contain all zeros or nans or infs

    while not good_map:
        # Normalize poiss DM norm to get correct counts/pix in ROI
        norm_gce = theta[0] / np.mean(temps_poiss[0][~mask_normalize_counts])

        # Grab the rest of the poiss norms
        norms_poiss = theta[1 : len(temps_poiss)]

        # Normalize PS map to get correct counts/pix in ROI
        # and construct appropriate dnds arrays for each PS template

        dnds_ary = []
        idx_theta_ps = len(temps_poiss)
        for idx, temp_ps in enumerate(temps_ps):
            dnds_ary_temp = dnds_list[idx]
            s_exp = np.trapz(s_ary * dnds_ary_temp, s_ary)
            temp_ratio = np.sum(temp_ps[~mask_normalize_counts]) / np.sum(temp_ps)
            exp_ratio = np.mean(exp_map[~mask_normalize_counts]) / np.mean(exp_map)
            dnds_ary_temp *= theta[idx_theta_ps] * np.sum(~mask_normalize_counts) / s_exp / temp_ratio / exp_ratio
            dnds_ary.append(dnds_ary_temp)
            idx_theta_ps += 1

        exp_map_norm = exp_map / np.mean(exp_map)  #  * exp_ratio

        # Draw PSs and simulate map

        sm = SimulateMap(temps_poiss, [norm_gce] + list(norms_poiss), [s_ary] * len(temps_ps), dnds_ary, temps_ps, psf_r_func, exp_map_norm, mask_roi=mask_roi)

        the_map_temp = sm.create_map()

        the_map_temp[mask_roi] = 0.0
        the_map = the_map_temp[~mask_sim].astype(np.float32)

        # Grab auxiliary variables
        mean_map = np.mean(the_map)
        var_map = np.var(the_map)

        the_map = the_map.reshape((1, -1))
        aux_vars = np.array([np.log(mean_map), np.log(np.sqrt(var_map))]).reshape((1, -1))

        # Resimulate if map is crap
        if (np.sum(the_map) == 0) or np.sum(np.isnan(the_map)) or np.sum(np.isinf(the_map)):
            good_map = False
            logger.info("Resimulating a crap map...")
        else:
            good_map = True

    return (the_map, aux_vars)
