# Import standard library modules
import os

# Import third-party modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import h5py
from bmadx import  Drift, SBend, Quadrupole, Sextupole
from tqdm import tqdm

# Import modules specific to this package
from utility_functions import check_input_consistency, isotime
from yaml_parser import parse_yaml
from attribute_initializer import init_attributes
from lattice import Lattice
from beam import Beam
from df_tracker import DF_Tracker
from interpolation import *

"""
Module Name: csr2d.py

Contains the CSR2D class which is the controller class for CSR wake computations
"""

class CSR2D:
    """
    Class to compute 2D CSR wake, uses adapative meshes at each time step to best fit the beam distribution
    """
    def __init__(self, input_file, parallel = False):
        """
        Creates an instance of the CSR3D class
        Parameters:
            input_file: assumed to be a configuration in YMAL file format that can be converted to dict
            parallel: boolean, indicate if parallel processing should be used for computations
        Returns:
            instance of CSR3D
        """
        # Convert YAML data into input dictionary
        self.input_dict = parse_yaml(input_file)

        # Check that input file is in the correct format
        check_input_consistency(self.input_dict)

        # Create the statistics and parameters dictionary along with the 3 main class instances
        (self.statistics,
         self.integration_params,
         self.csr_params,
         self.lattice,
         self.beam,
         self.df_tracker,
         self.wake_history) = init_attributes(self.input_dict)

        # charge in C (8.98755e-6 MeV/m for 1nC/m^2)
        self.CSR_scaling = 8.98755e3 * self.beam.charge 

        # Get the current time
        self.timestamp = isotime()

        # Create the prefix (for naming)
        self.prefix = f'{self.csr_params["write_name"]}-{self.timestamp}'

        # If we are using parallel computing
        if parallel:
            self.parallel = True
        else:
            self.parallel = False

        print("CSR2D Object Succesfully Initialized")

    def run(self):
        """
        Propagates the beam through the inputed lattice, calculating and appling CSR wake if requested by the user
        """
        # Populate the df_tracker with the first step and update the statistics
        self.df_tracker.populate_step(0, self.beam)
        self.update_statistics(0)
        
        # Starting at the second step, we propagate, populate, compute CSR, and apply CSR
        #for step_index in tqdm(range(1, self.df_tracker.total_steps)):
        for step_index in range(1,2):
            # Propagate the beam to the current step position
            self.beam.track(self.lattice.bmadx_elements[step_index-1])

            # Populate the  with the beam distribution
            csr_params, t2_csr, R_inv_csr, CSR_mesh = self.df_tracker.populate_step(step_index, self.beam)

            # Compute the formation length of the beam at this moment in time
            formation_length = self.get_formation_length(step_index)

            # Check to see if we should compute csr at this step
            if self.lattice.compute_CSR[step_index] and self.csr_params["compute_CSR"]:

                # Compute CSR wake on the mesh and write the wake values to their dictionary
                dE_vals, x_kick_vals = self.compute_CSR_on_mesh(CSR_mesh, formation_length, step_index)
                self.wake_history["dE_history"][step_index-1] = dE_vals
                self.wake_history["x_kick_history"][step_index-1] = x_kick_vals
        
                # The distance to the last step at which the csr wake was computed
                d_step = self.lattice.step_size*self.lattice.steps_since_CSR_applied[step_index]

                # Apply CSR wakes to beam
                if self.csr_params["apply_CSR"]:
                    p_sd, o_sd = self.df_tracker.p_sd[step_index], self.df_tracker.o_sd[step_index]
                    self.beam.apply_wakes(dE_vals, x_kick_vals, csr_params, t2_csr, R_inv_csr, d_step, p_sd, o_sd)

                # Populate the df_tracker with the new beam distribution, do not update the CSR_mesh
                self.df_tracker.populate_step(step_index, self.beam, update_CSR_mesh = False)

            # Dump the beam at this step if desired by the user
            if (step_index+1) in self.csr_params["write_beam"]:
                self.dump_beam(directory="/Users/treyfischbach/Desktop/SULI 2024/Wake Benchmarking/data")

            # Update the statistics dict
            self.update_statistics(step_index)

    # TODO: fix the case for non dipole elements
    def get_formation_length(self, step_index):
        """
        Computes the formation length for the beam at inputed step referenece trajectory and current beam shape.
        Note that we use the formation length of the element in which the prvious step was in
        """
        # Index the previous element
        previous_bmadx_element = self.lattice.bmadx_elements[step_index-1][-1]

        # Check to see if it is a dipole or not
        if isinstance(previous_bmadx_element, SBend):
            R = 1/(previous_bmadx_element.G)
        
        else:
            R = 1.0

        return 2*self.lattice.step_size

    def compute_CSR_on_mesh(self, CSR_mesh, formation_length, step_index):
        """
        Computes the CSR wake at each point in the CSR_mesh
        Parameters:
            CSR_mesh: the mesh on which each vertex the CSR_wake is to be computed
            formation_length: the formation_length of the beam at this step
        """
        # Flatten the CSR_mesh
        Z = CSR_mesh[:,:,1].flatten()
        X = CSR_mesh[:,:,0].flatten()

        # Define the arrays where the wake values will be stored
        dE_vals = np.zeros(len(Z))
        x_kick_vals = np.zeros(len(Z))

        # Precompute these beam characteristic here
        slope = np.tan(self.df_tracker.tilt_angle[step_index])
        x_mean = self.beam.mean_x
        x_std = self.beam.sigma_x
        z_std = self.beam.sigma_z

        # For each point on the CSR_mesh, compute dE and x_kick and update their respective arrays
        for index in range(len(X)):
            s = self.beam.position + Z[index]
            x = X[index]

            dE_vals[index], x_kick_vals[index] = self.compute_CSR_at_point(s, x, formation_length, slope, x_mean, x_std, z_std, step_index)

        # Reshape dE_vals and x_kick_vals to be the dimension of the mesh again
        original_shape = CSR_mesh.shape[:2]
        dE_vals = dE_vals.reshape(original_shape)
        x_kick_vals = x_kick_vals.reshape(original_shape)

        return dE_vals, x_kick_vals

    def compute_CSR_at_point(self, s, x, formation_length, slope, x_mean, x_std, z_std, step_index):
        """
        Helper function to compute_CSR_on_mesh, computes the CSR wake at a singular point on the CSR mesh
        Parameters:
            s and x: position of point on which we want to compute CSR wake
            formation_length: the formation_length of the beam at this step
            slope: the x/z slope of the beam distribution
            x_mean, x_std, z_std: some beam characteristics
            step_index: the index of the current step
        Returns:
            dE, x_kick: the longitudinal and transverse CSR wake
        """
        # Compute the integration areas
        integration_areas = self.get_integration_areas(s, x, formation_length, slope, x_mean, x_std, z_std)

        # Initialize the dE and x_kick
        dE = 0
        x_kick = 0

        # Compute the integrand over each area, integrate, and then sum the contribution
        for area in integration_areas:
            integrand_z, integrand_x = self.get_CSR_integrand(s, x, area)

            # Integrate over these real quick using trap
            dE += -self.CSR_scaling * np.trapz(y=np.trapz(y=integrand_z, x=area[0][:, 0], axis=0), x=area[1][0,:])
            x_kick += self.CSR_scaling * np.trapz(y=np.trapz(y=integrand_x, x=area[0][:, 0], axis=0), x=area[1][0,:])

        return dE, x_kick
    
    def get_integration_areas(self, s, x, formation_length, slope, x_mean, x_std, z_std):
        """
        Helper function to compute_CSR_at_point. Given a point on the CSR mesh, computes the three or 
        four areas of space in the lab frame over which we must integrate
        Parameters:
            s and x: position of point on which we want to compute CSR wake
            formation_length: the formation length of the beam at the current step
            slope: the x/z slope of the beam distribution
            x_mean, x_std, z_std: some beam characteristics
        Returns:
            integration_areas: an array containing [x_mesh, s_mesh] meshgrids for each integration area
        """
        x0 = (s - self.beam.position)*slope

        # Boolean tracking if the beam is sufficently tilted
        sufficent_tilt = np.abs(slope) > 1

        # If the band is sufficently tilted, then the integration region changes
        if sufficent_tilt:

            # Construct the integration regions in a way that depends on  which way the band is tiled
            if slope > 0:
                tan_alpha = -2 * slope / (1 - slope ** 2)  # alpha = pi - 2 theta, tan_alpha > 0
                d = (10 * x_std + x_mean - x) / tan_alpha

                s4 = s + 3 * z_std
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * z_std

                # area 1
                x1_l = x + 0.1 * x_std
                x1_r = x + 10 * x_std

                # area 2
                x2_l = x - 3 * x_std
                x2_r = x1_l

                # area 3
                x3_l = x0 - 5 * x_std
                x3_r = x0 + 5 * x_std

                x4_l = x0 - 20 * x_std
                x4_r = x0 + 20 * x_std

            else:
                tan_alpha = 2 * slope / (1 - slope ** 2)
                d = -(x_mean - x - 10 * x_std) / tan_alpha

                s4 = s + 3 * z_std
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * z_std

                # area 1
                x1_l = x - 10 * x_std
                x1_r = x - 1 * x_std

                # area 2
                x2_l = x1_r
                x2_r = x + 3 *x_std

                # area 3
                x3_l = x0 - 5 * x_std
                x3_r = x0 + 5 * x_std

                x4_l = x0 - 20 * x_std
                x4_r = x0 + 20 * x_std

            # Make sure that we cap the smallest intergration bound at s=0, where the nominal path begins
            s1 = np.max((0, s2 - self.integration_params["n_formation_length"] * formation_length))
            
            sp1 = np.linspace(s1, s2, self.integration_params["zbins"])
            sp2 = np.linspace(s2, s3, self.integration_params["zbins"])
            sp3 = np.linspace(s3, s4, self.integration_params["zbins"])
            xp1 = np.linspace(x1_l, x1_r, self.integration_params["xbins"])
            xp2 = np.linspace(x2_l, x2_r, self.integration_params["xbins"])
            xp3 = np.linspace(x3_l, x3_r, self.integration_params["xbins"])
            xp4 = np.linspace(x4_l, x4_r, 2*self.integration_params["xbins"])

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp4, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp3, sp2, indexing = 'ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp1, sp3, indexing='ij')
            [xp_mesh4, sp_mesh4] = np.meshgrid(xp2, sp3, indexing='ij')
            integration_areas = [[xp_mesh1, sp_mesh1], [xp_mesh2, sp_mesh2], [xp_mesh3, sp_mesh3], [xp_mesh4, sp_mesh4]]

        else:
            s2 = s - 500 * z_std
            s3 = s - 20 *z_std
            s4 = s + 5 * z_std

            x1_w = x0 - 20 * x_std
            x2_w = x0 + 20 * x_std

            x1_n = x0 - 10 * x_std
            x2_n = x0 + 10 * x_std

            # Make sure that we cap the smallest intergration bound at s=0, where the nominal path begins
            s1 = np.max((0, s2 - self.integration_params["n_formation_length"] * formation_length))

            # Initalize the integration meshes with the desired number of bins
            sp1 = np.linspace(s1, s2, self.integration_params["zbins"])
            sp2 = np.linspace(s2, s3, self.integration_params["zbins"])
            sp3 = np.linspace(s3, s4, self.integration_params["zbins"])
            xp_w = np.linspace(x1_w, x2_w, 2*self.integration_params["xbins"])
            xp_n = np.linspace(x1_n, x2_n, self.integration_params["xbins"])

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp_w, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp_n, sp2, indexing='ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp_n, sp3, indexing='ij')
            integration_areas = [[xp_mesh1, sp_mesh1], [xp_mesh2, sp_mesh2], [xp_mesh3, sp_mesh3]]

        return integration_areas

    def get_CSR_integrand(self, s, x, area):
        """
        Helper function to compute_CSR_at_point, finds the integrand contribution (W1 + W2 + W3) of the inputed integration area
        to the specific point on the CSR mesh
        Parameters:
            s and x: position of point on CSR mesh at which we want to compute CSR wake
            area: integration meshgrid
        Returns:
            CSR_integrand_z and CSR_integrand_x
        """
        # Define t
        t = self.beam.position

        # Flatten the meshgrid
        sp_flat = area[1].ravel()
        xp_flat = area[0].ravel()

        # Get x & x prime and y & y prime in the lab frame
        # the 'prime' variables are the integration mesh grid converted to lab frame coordinates
        X0_s = interpolate1D(xval = np.array([s]), data = self.lattice.lab_frame_sample_coords[:, 0], min_x = self.lattice.min_s,
                             delta_x = self.lattice.delta_s)[0]
        X0_sp = interpolate1D(xval = sp_flat, data = self.lattice.lab_frame_sample_coords[:, 0], min_x = self.lattice.min_s,
                              delta_x = self.lattice.delta_s)
        Y0_s = interpolate1D(xval = np.array([s]), data = self.lattice.lab_frame_sample_coords[:, 1], min_x = self.lattice.min_s,
                             delta_x = self.lattice.delta_s)[0]
        Y0_sp = interpolate1D(xval = sp_flat, data = self.lattice.lab_frame_sample_coords[:, 1], min_x = self.lattice.min_s,
                              delta_x = self.lattice.delta_s)

        # Do the same for the normal and tangential vectors
        n_vec_s_x = interpolate1D(xval = np.array([s]), data = self.lattice.sample_n_vecs[:, 0], min_x = self.lattice.min_s,
                                  delta_x = self.lattice.delta_s)[0]
        n_vec_sp_x =interpolate1D(xval = sp_flat, data = self.lattice.sample_n_vecs[:, 0], min_x = self.lattice.min_s,
                                  delta_x = self.lattice.delta_s)
        n_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.sample_n_vecs[:, 1], min_x=self.lattice.min_s,
                                  delta_x=self.lattice.delta_s)[0]
        n_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.sample_n_vecs[:, 1], min_x=self.lattice.min_s,
                                   delta_x=self.lattice.delta_s)
        tau_vec_s_x = interpolate1D(xval=np.array([s]), data=self.lattice.sample_tau_vecs[:, 0], min_x=self.lattice.min_s,
                                  delta_x=self.lattice.delta_s)[0]
        tau_vec_sp_x = interpolate1D(xval=sp_flat, data=self.lattice.sample_tau_vecs[:, 0], min_x=self.lattice.min_s,
                                   delta_x=self.lattice.delta_s)
        tau_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.sample_tau_vecs[:, 1], min_x=self.lattice.min_s,
                                  delta_x=self.lattice.delta_s)[0]
        tau_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.sample_tau_vecs[:, 1], min_x=self.lattice.min_s,
                                   delta_x=self.lattice.delta_s)

        # Comupte the magnitude of the different between r and rp (r prime) vectors
        r_minus_rp_x = (X0_s + x * n_vec_s_x) - (X0_sp + xp_flat * n_vec_sp_x)      # r-r' in the X coordinate of the lab frame
        r_minus_rp_y = (Y0_s + x * n_vec_s_y) - (Y0_sp + xp_flat * n_vec_sp_y)      # r-r' in the Y coordinate of the lab frame
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)

        # Retarded time array for each point in meshgrid
        t_ret = t - r_minus_rp

        # The effective radius from the lab frame origin for each lattice step
        rho_sp = np.zeros(sp_flat.shape)

        # Populate rho_sp
        for count in range(self.lattice.element_num):
            if count == 0:
                rho_sp[sp_flat < self.lattice.element_distances[count]] = self.lattice.element_rho_vals[count]
            else:
                rho_sp[(sp_flat < self.lattice.element_distances[count]) & (sp_flat >= self.lattice.element_distances[count - 1])] = self.lattice.element_rho_vals[count]

        # Interpolate the velocity of the beam at the current point
        translated_point = translate_points(np.array([t]), np.array([s-t]), np.array([x]),
                                             self.df_tracker.t1_h, self.df_tracker.C_inv_h, self.df_tracker.R_inv_h, self.df_tracker.t2_h,
                                             self.lattice.step_ranges, np.array([0]), np.zeros((1, 5)))
        
        beta_x = interpolate3D(translated_point, [self.df_tracker.beta_x], self.lattice.step_size, np.zeros((5, 1)))[0][0]

        # Initialize translated mesh points and the index of each point
        translated_points = np.zeros((len(t_ret), 5))
        p_indices = np.arange(len(t_ret), dtype=int)

        # Populate the translated mesh points with all the points on the mesh
        translated_points = translate_points(t_ret, sp_flat - t_ret, xp_flat,
                                             self.df_tracker.t1_h, self.df_tracker.C_inv_h, self.df_tracker.R_inv_h, self.df_tracker.t2_h,
                                             self.lattice.step_ranges, p_indices, translated_points)

        # Interpolate retarted time quantites!
        interp_result = np.zeros((5, len(t_ret)), dtype=np.float64)
        interp_data = (self.df_tracker.density, 
                       self.df_tracker.beta_x, 
                       self.df_tracker.partial_density_x, 
                       self.df_tracker.partial_density_z,
                       self.df_tracker.partial_beta_x)
        
        interp_result = interpolate3D(translated_points, interp_data, self.lattice.step_size, interp_result)

        # Unpack the interpolate result
        density_ret = interp_result[0]
        beta_x_ret = interp_result[1]
        partial_density_x_ret = interp_result[2]
        partial_density_z_ret = interp_result[3]
        partial_beta_x_ret = interp_result[4]

        # TODO: More accurate vx, maybe add vs
        beta_s = 1
        beta_s_ret = 1            # beta_s, set to 1 as we assume velocity in s direction to = c
        partial_beta_s_ret = 0    # Partial beta_s_ret wrt to s

        # Accounts for transfer to the lab frame
        scale_term =  1 + xp_flat*rho_sp

        # LONGITUDINAL WAKE
        ############################################
        # Compute beta in the lab frame for current time and retarded
        beta_lab_x = beta_s * tau_vec_s_x + beta_x * n_vec_s_x
        beta_lab_y = beta_s * tau_vec_s_y + beta_x * n_vec_s_y
        beta_ret_lab_x = beta_s_ret * tau_vec_sp_x + beta_x_ret * n_vec_sp_x
        beta_ret_lab_y = beta_s_ret * tau_vec_sp_y + beta_x_ret * n_vec_sp_y

        # Dot product of beta with beta_ret using lab frame as coordinate systen
        beta_dot_beta_ret = beta_lab_x * beta_ret_lab_x + beta_lab_y * beta_ret_lab_y

        # "constants 1x and 2y"
        C1x = beta_lab_x - beta_dot_beta_ret * beta_ret_lab_x
        C1y = beta_lab_y - beta_dot_beta_ret * beta_ret_lab_y

        # Partial Density wrt x lab and y lab
        partial_density_ret_lab_x = partial_density_x_ret  * n_vec_sp_x + partial_density_z_ret / scale_term * tau_vec_sp_x
        partial_density_ret_lab_y = partial_density_x_ret * n_vec_sp_y + partial_density_z_ret / scale_term * tau_vec_sp_y

        # "Integrand of 1st integral in the z wake"
        Iz1 =  scale_term * (C1x*partial_density_ret_lab_x + C1y*partial_density_ret_lab_y)

        # Divergence of velocity
        divergence_velocity = partial_beta_s_ret + partial_beta_x_ret

        # "Integrand of 2nd integral in the z wake"
        Iz2 = scale_term * beta_dot_beta_ret * density_ret * divergence_velocity

        # Longitudinal wake integrand, note that the sign here is flipped from the derivations
        CSR_integrand_z = Iz1/r_minus_rp - Iz2/r_minus_rp
        CSR_integrand_z = partial_density_ret_lab_x + partial_density_ret_lab_y
        
        ############################################

        # TRANSVERSE WAKE - X
        ############################################
        n_minus_np_x = n_vec_s_x - n_vec_sp_x       # n-n' in the X coordinate of the lab frame
        n_minus_np_y = n_vec_s_y - n_vec_sp_y       # n-n' in the Y coordinate of the lab frame

        # part1: (r-r')(n - n')
        part1 = r_minus_rp_x * n_minus_np_x + r_minus_rp_y * n_minus_np_y

        #part2: n tau'
        part2 = n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y

        # part3: partial density/partial t_ret
        partial_density = - (beta_ret_lab_x * partial_density_ret_lab_x + beta_ret_lab_y * partial_density_ret_lab_y) - \
                          density_ret * divergence_velocity

        # Three integrands for transverse wake
        W1 = (scale_term * part1 * density_ret) / (r_minus_rp * r_minus_rp * r_minus_rp)
        W2 = (scale_term * part1 * partial_density) / (r_minus_rp * r_minus_rp)
        W3 = -(scale_term * part2 * partial_density) / r_minus_rp

        CSR_integrand_x = W1 + W2 + W3
        CSR_integrand_x = CSR_integrand_x.reshape(area[1].shape)
        CSR_integrand_z = CSR_integrand_z.reshape(area[1].shape)

        return CSR_integrand_z, CSR_integrand_x

    def update_statistics(self, step):
        """
        Updates the statistics dictionary with the current step's beam characteristics
        """
        twiss = self.beam.twiss
        self.statistics['twiss']['alpha_x'][step] = twiss['alpha_x']
        self.statistics['twiss']['beta_x'][step] = twiss['beta_x']
        self.statistics['twiss']['gamma_x'][step] = twiss['gamma_x']
        self.statistics['twiss']['emit_x'][step] = twiss['emit_x']
        self.statistics['twiss']['eta_x'][step] = twiss['eta_x']
        self.statistics['twiss']['etap_x'][step] = twiss['etap_x']
        self.statistics['twiss']['norm_emit_x'][step] = twiss['norm_emit_x']
        self.statistics['twiss']['alpha_y'][step] = twiss['alpha_y']
        self.statistics['twiss']['beta_y'][step] = twiss['beta_y']
        self.statistics['twiss']['gamma_y'][step] = twiss['gamma_y']
        self.statistics['twiss']['emit_y'][step] = twiss['emit_y']
        self.statistics['twiss']['eta_y'][step] = twiss['eta_y']
        self.statistics['twiss']['etap_y'][step] = twiss['etap_y']
        self.statistics['twiss']['norm_emit_y'][step] = twiss['norm_emit_y']
        self.statistics['slope'][step] = np.tan(self.df_tracker.tilt_angle[step])
        self.statistics['sigma_x'][step] = self.beam._sigma_x
        self.statistics['sigma_z'][step] = self.beam._sigma_z
        self.statistics['sigma_energy'][step] = self.beam.sigma_energy
        self.statistics['mean_x'][step] = self.beam._mean_x
        self.statistics['mean_z'][step] = self.beam._mean_z
        self.statistics['mean_energy'][step] = self.beam.mean_energy

    def save_data(self, directory="", filename=""):
        """
        Once simulation is complete we save the data into an h5 file
        """
        if not filename:
            filename = f"{self.prefix}"
    
        if not directory:
            directory = '.'  # Default to current directory if no directory is specified

        filename_stats = filename+"_stats.h5"
        filename_wakes = filename+"_wakes.h5"
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Construct the full file path for the stats and wakes
        file_path_stats = os.path.join(directory, filename_stats)
        file_path_wakes = os.path.join(directory, filename_wakes)

        # Save statistics dictionary to an HDF5 file
        with h5py.File(file_path_stats, 'w') as f:

            # statistics
            stats_grp = f.create_group('statistics')
            for key, value in self.statistics.items():
                if isinstance(value, dict):
                    sub_grp = stats_grp.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_grp.create_dataset(sub_key, data=sub_value)
                else:
                    stats_grp.create_dataset(key, data=value)
        
        # Save wakes (if requested) to a HDF5 file
        if self.csr_params["write_wakes"]:
            with h5py.File(file_path_wakes, 'w') as f:
                groupname = "wake_history"
                g = f.create_group(groupname)
                g.create_dataset("csr_mesh_coords", data=self.df_tracker.csr_coords[1:])
                g.create_dataset("dE_history", data=self.wake_history["dE_history"])
                g.create_dataset("x_kick_history", data=self.wake_history["x_kick_history"])
            
    def dump_beam(self, directory="", filename=""):
        """
        Record the current beam in the particle_group format
        """
        if not filename:
            filename = str(self.beam.step) + ".h5"
    
        if not directory:
            directory = '.'  # Default to current directory if no directory is specified
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")

        print("Beam at position {} is written to {}".format(self.beam.position, filename))

        self.beam.particle_group.write(file_path)


