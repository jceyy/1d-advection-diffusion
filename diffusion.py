import numpy as np
import time
import os
import matplotlib.pyplot as plt
import scipy.fftpack
from mpi4py import MPI
from dedalus import public as de
# from dedalus.extras.plot_tools import quad_mesh, pad_limits
from dedalus.extras import flow_tools
from os.path import isfile, join
from dedalus.tools import post
import h5py
import sys
# import itertools
import logging
import shutil
import bchandler

from dedalus_interface import DedalusInterface

logger = logging.getLogger(__name__)

plt.rcParams.update({'font.size': 36})
plt.rcParams["figure.figsize"] = (24, 18)


class DedalusPy(DedalusInterface):
    default_muName = 'None'

    def __init__(self, argv=None):
        super(DedalusPy, self).__init__()

        self.nx = [128]
        self.xlim = [[0, 1]]
        self.D = 1
        self.a = 2

        self.dt = 1e-1
        self.t = 20
        self.wt = 3600
        self.niter = 10000

        self.da = 3 / 2
        self.nfields = len(self.nx)
        self.discrete_sym = 'None'

        self.bc_handle = bchandler.BcHandler(1, "x")
        self.bc_handle.set("x", [(0, 1), (1, +1)])
        self.x_nsolver = np.zeros((self.nx[0] - 2) * self.nfields)
        self.nfields = 1
        self.err_steady_state_u = []

        self.L = [self.xlim[0][1] - self.xlim[0][0]]
        self.x = np.zeros(self.nx[0])
        self.x_basis = 0
        self.domain = 0
        self.problem = 0
        self.solver = 0
        self.kx = 0
        self.x_scaled = 0
        self.u = np.zeros(self.nx[0])
        self.ux = np.zeros(self.nx[0])

        self.export_data = False
        self.r = False

        self.ts = de.timesteppers.RK443
        self.domain_setup()
        self.problem_setup()
        self.build_solver()
        self.init_problem()
        myprint("----------- Dedalus setup complete --------------\n")

    def domain_setup(self):
        # Bases and domain
        self.x_basis = de.Chebyshev('x', self.nx[0], interval=(self.xlim[0][0], self.xlim[0][1]), dealias=self.da)
        self.domain = de.Domain([self.x_basis], grid_dtype=np.float64)

    def problem_setup(self, mu=-1, muName='none'):
        if muName == 'L':
            # Rebuild domain
            self.L[0] = mu
            self.domain_setup()

        if muName != 'none':
            self.problem.parameters[muName] = mu  # Either Ra, Ta, Pr, Ty, Ly, phi

        if muName == 'D':
            self.D = mu
        elif muName == 'a':
            self.a = mu

        # Problem
        self.problem = de.IVP(self.domain, variables=['u', 'ux'])
        self.problem.parameters['D'] = self.D
        self.problem.parameters['a'] = self.a

        # Update dependent parameters
        self.problem.add_equation("dt(u) - D * dx(ux) = -a * u*ux")
        self.problem.add_equation("ux - dx(u) = 0")

        # Boundary conditions
        print("Dealing with BCs\n\n")
        bc_str = self.bc_handle.gen_bc_str()
        for bc in bc_str:
            print(bc)
            self.problem.add_bc(bc)
        return

    def build_solver(self):
        # Build solver
        self.solver = self.problem.build_solver(self.ts)
        self.solver.stop_wall_time = self.wt
        self.solver.stop_iteration = self.niter
        self.solver.stop_sim_time = self.t

        if self.export_data:
            pass
            # self.folder_exist_check()
            # snapshots = self.solver.evaluator.add_file_handler(self.snapshots_folder, max_writes=self.itmax,
            #                                              sim_dt=self.odt)
            # snapshots.add_system(self.solver.state, layout='g')
            # snapshots.add_task("u + DU*(z-0.5)", name="u_tot")
            #
            # energies = self.solver.evaluator.add_file_handler(self.energies_folder, max_writes=self.itmax, sim_dt=self.odt)
            # energies.add_task("integ(u*u+dy(psi)*dy(psi)+dz(psi)*dz(psi),'y','z')/(2*Ly*Lz)", name="KE")
            # energies.add_task("integ((u+DU*(z-0.5))*(u+DU*(z-0.5))+dy(psi)*dy(psi)+dz(psi)*dz(psi),'y','z')/(2*Ly*Lz)", name="KE_tot")
            # energies.add_task("integ(theta*theta,'y','z')/(Ly*Lz)", name="TH")
            # energies.add_task("integ(dz(theta),'y')/Ly", name="Tfluc z")
            # energies.add_task("integ((theta+1-z)*dy(-psi),'y','z')/(Ly*Lz)", name="wT")
            # energies.add_task("integ((integ(u,'y')/Ly)**2,'z')/Lz*0.5", name="KEubar")
            # energies.add_task("integ((integ(u+DU*(z-0.5),'y')/Ly)**2,'z')/Lz*0.5", name="KEutotbar")
            # energies.add_task("integ((integ(dz(psi),'y')/Ly)**2,'z')/Lz*0.5", name="KEvbar")
            # energies.add_task("(integ(u*u+dy(psi)*dy(psi)+dz(psi)*dz(psi),'y','z')/(Ly*Lz))**0.5", name="urms")
            # energies.add_task("(integ((u+DU*(z-0.5))*(u+DU*(z-0.5))+dy(psi)*dy(psi)+dz(psi)*dz(psi),'y','z')/(Ly*Lz))**0.5", name="u_totrms")
            #
            # profiles = self.solver.evaluator.add_file_handler(self.profiles_folder, max_writes=self.itmax, sim_dt=self.odt)
            # profiles.add_task("integ(theta,'y')/Ly", name="T")
            # profiles.add_task("integ(u,'y')/Ly", name="u")
            # profiles.add_task("integ(u + DU*(z-0.5),'y')/Ly", name="u_tot")
            # profiles.add_task("integ(dz(psi),'y')/Ly", name="v")
            # profiles.add_task("integ(-dy(psi),'y')/Ly", name="w")

    def init_problem(self):
        # Initial conditions
        self.x = self.domain.grid(0)
        self.kx = self.domain.elements(0)
        self.x_scaled = self.domain.grid(0, self.da)
        self.u = self.solver.state['u']
        self.ux = self.solver.state['ux']

        amp0 = 1e-6
        self.u['g'] = amp0 * np.sin(2 * np.pi * self.x / self.L[0])
        self.update_diff_var()

    def shifts(self, ay, az):
        # print("\t\t********** shifts ********** (does nothing)")
        # self.u['c'] = self.u['c'] * np.exp(self.ky * 1j * ay + self.kz * 1j * az)
        return

    def update_diff_var(self):
        self.u.differentiate('x', out=self.ux)

    def add_perturbations(self, mag, decay):
        print("\t\t********** add_perturbations ********** (does nothing)")
        return

    def set_scales(self, s):
        self.u.set_scales(s)
        self.ux.set_scales(s)

    def observable(self, x):
        self.set_scales(1)
        self.to_fields(x)

        obs = -self.D * de.operators.differentiate(self.u, 'x')
        obs = obs.evaluate()
        obs.set_scales(1)

        obs = obs['g']  # [int(self.nx[0]/2)]

        print("j = " + str(obs))
        return obs

    def diff(self, u_init, direction):  # Still need to check
        print("\t\t********** diff ********** NOT PROPERLY CHECKED YET")
        return None

    def advance(self, T, u_init=None):
        # print("\t\t********** advance **********")
        init_time = self.solver.sim_time
        input_size = len(self.x_nsolver)
        self.set_scales(1)

        # print('Input size %d' % input_size)
        # print('Len u_init %d' % len(u_init))
        # if self.travelling:  # Adding symmetries
        #     input_size = input_size + 1
        #     if self.yrel:
        #         input_size = input_size + 1

        # print('%d. vector length:  \t data to read %d \t intput size: %d\n' %
        #       (len(u_init), self.ny*(self.nz-2)*self.nfields, input_size))
        if u_init is not None:
            if len(u_init) == input_size:
                # myprint("Reading input array in Python of length: " + str(len(u_init)))
                myprint('')
                MPI.COMM_WORLD.Barrier()
                self.to_fields(u_init)
            else:
                myprint("Warning: size mismatch from python API")

        if T == np.nan or T == np.inf or -T == np.inf:
            myprint('Warning: T is nan, returning u_init')
            return u_init
        step_size = self.dt if self.dt <= T else T
        while self.solver.sim_time + step_size < init_time + T:
            self.solver.step(step_size)
            self.print_log(10, False)
            flux, sstate = self.get_flux__steady_state()
            self.u.set_scales(1)
            u = self.u['g']
            self.err_steady_state_u.append(np.mean(np.abs(u - sstate)) / np.mean(np.abs(u)))
            # self.plot_intermediate()

        self.solver.step(T - (self.solver.sim_time - init_time))
        self.print_log(10, False)
        self.symmetry()
        self.set_scales(1)

        if u_init is None:
            myprint('u_init is None!')

        self.to_vector()

        myprint('')  # Give some space to the output in the terminal
        MPI.COMM_WORLD.Barrier()
        # print('End advance\n\n\n')
        return self.x_nsolver

    def get_flux__steady_state(self):
        flux = -self.D * de.operators.differentiate(self.u, 'x')
        flux = flux.evaluate()
        flux.set_scales(1)

        if self.a == 0:
            sstate = np.zeros(self.nx[0])
        else:
            sstate = self.D / self.a * de.operators.differentiate(de.operators.differentiate(self.u, 'x'), 'x') / \
                 de.operators.differentiate(self.u, 'x')
            sstate = sstate.evaluate()
            sstate.set_scales(1)
            sstate = sstate['g']
        return flux['g'], sstate

    def plot_intermediate(self):
        plt.figure()
        flux, sstate = self.get_flux__steady_state()
        self.u.set_scales(1)
        plt.plot(self.x, self.u['g'])
        plt.plot(self.x, flux)
        if self.a != 0:
            plt.plot(self.x, sstate, '--')
        plt.xlabel('$x$')
        plt.legend(['$u$', '$j = -D \\partial_x u$', '$u^* = D/a\\, \\partial_x^2 u / \\partial_x u}$'])
        plt.grid()
        plt.title("$t = %.2f, \\;\\partial_t u + a u\\partial_x u = D\\partial^2_x u, \\;\\; D = %.3f, \\, a=%.3f$" %
                  (self.solver.sim_time, self.D, self.a))
        plt.show(block=True)

    def to_vector(self):  # Does not take into account symmetries (yet)!
        # print("\t\t********** to_vector **********")
        self.x_nsolver = self.u['g'][1:-1]
        return

    def to_fields(self, vector):
        # print("\t\t********** to_fields **********")
        self.u['g'][1:-1] = vector

        # BC reconstruction process
        # for now only dirichlet
        self.u['g'][0] = self.bc[0][0][1]
        self.u['g'][-1] = self.bc[0][-1][1]

        self.update_diff_var()
        return

    def updateMu(self, mu=-1, muName='None'):
        print("\t\t********** updateMu ********** (work)")
        # print("%s updated to $.6f" % (muName, mu))
        if muName == "None":
            muName = self.default_muName  # Why here?

        if muName in self.problem.parameters:
            if muName == 'L':
                self.L[0] = mu
            self.domain_setup()  # depends on L
            self.problem_setup(mu, muName)
            self.build_solver()
            self.init_problem()
            myprint("Updating " + muName + " to: " + str(mu))
        else:
            raise Exception("Incorrect string provided for mu name\n")
        return

    def read_h5(self, filename, n_it):
        # myprint("\t\t********** read_h5 **********")

        self.solver.load_state(filename, n_it)
        self.solver.iteration = 0
        self.solver.sim_time = 0
        self.solver.dt = self.dt
        self.set_scales(1)
        u_out_ = self.to_vector()

        myprint('Loaded successfully from file %s' % filename)
        return u_out_

    def symmetry(self):
        # print("\t\t********** symmetry ********** (does nothing)")
        if self.discrete_sym == 'None':
            return

        elif self.discrete_sym == 'TW':
            self.set_scales(1)
            self.v['g'] = (self.v['g'] + np.transpose(self.u['g'])) / 2
            self.u['g'] = np.transpose(self.v['g'])

        elif self.discrete_sym == 'FP':
            self.u.set_scales(1)
            self.v.set_scales(1)

            self.u['g'] = (self.u['g'] - np.flip(self.u['g'])) / 2
            self.v['g'] = (self.v['g'] + np.flip(self.v['g'])) / 2
        return

    def folder_exist_check(self):
        if MPI.COMM_WORLD.Get_rank() == 0 and self.r is False:
            # Check if _r or _s is present in the output file name
            if self.output_dir.find('_r') != -1 or self.output_dir.find('_s') != -1:
                print('The characters sequences \'_r\' and \'_s\' are forbidden from the output string'
                      ' name (you set \'%s\')' % self.output_dir)
                exit(1)
            output_str = os.path.join(self.working_directory, self.output_dir)
            if os.path.isdir(output_str):
                print('Warning: the folder \'%s\' already exists. Do you want to continue? (y/n)' % output_str)
                in_str = input('Note that if you do, all existing data inside will be erased.\n')
                print('')
                if in_str != 'y':  # If the user does not want ton proceed, exit
                    print('Quitting...\n')
                    exit(1)

                # 'y' pressed
                shutil.rmtree(output_str)
                os.mkdir(output_str)

            else:
                os.mkdir(output_str)

        elif self.r:
            self.snapshots_folder = self.snapshots_folder + '_r'
            self.energies_folder = self.energies_folder + '_r'
            self.profiles_folder = self.profiles_folder + '_r'
        return

    def merge_back_files(self):
        if self.export_data:
            # Merge different processes' files together
            post.merge_process_files(self.snapshots_folder, cleanup=True)
            post.merge_process_files(self.energies_folder, cleanup=True)
            post.merge_process_files(self.profiles_folder, cleanup=True)

        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.export_data:
                path = os.path.join(self.working_directory, self.o)

                ref_file = sorted([f for f in os.listdir(path)
                                   if isfile(os.path.join(path, f)) and f.split('.')[-1] == 'txt'])
                ref_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))

                start_number = 0
                if not self.r:
                    start_number = 0
                elif len(ref_file) != 0:
                    start_number = ref_file[-1].split('_s')[-1]  # extension still attached, for instance '5.h5'
                    start_number = int(start_number.split('.')[0])
                    # base = ref_file[-1].split('_s')[0]

                txt_file_path = os.path.join(self.working_directory, self.o, self.o + '_s' + str(start_number + 1) + '.txt')
                self.hdl.write(sys.argv, txt_file_path)

            # If restart : rename and move all the files from te restarted folder to the base one, and then delete it
            if self.r:
                folders_to_process = [self.snapshots_folder, self.energies_folder, self.profiles_folder]
                # Get rid of _r at the end of restarted folders
                folders_base = [self.snapshots_folder[0:-2], self.energies_folder[0:-2], self.profiles_folder[0:-2]]
                for i in range(len(folders_to_process)):
                    # Files to rename and copy
                    path = os.path.join(self.working_directory, folders_to_process[i])
                    txt_files = sorted([f for f in os.listdir(path)
                                       if isfile(os.path.join(path, f)) and f.split('.')[-1] == 'txt'])
                    h5_files = sorted([f for f in os.listdir(path)
                                       if isfile(os.path.join(path, f)) and f.split('.')[-1] == 'h5'])
                    txt_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))
                    h5_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))
                    # Reference file to get number (last h5 file)
                    path = os.path.join(self.working_directory, folders_base[i])

                    ref_file = sorted([f for f in os.listdir(path)
                                       if isfile(os.path.join(path, f)) and f.split('.')[-1] == 'h5'])
                    ref_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))
                    ref_file = ref_file[-1]

                    start_number = ref_file.split('_s')[-1]  # extension still attached, for instance '5.h5'
                    start_number = int(start_number.split('.')[0])
                    base = h5_files[-1].split('_r_s')[0]

                    # Rename and move h5 files from X_r to X folder
                    for j in range(len(h5_files)):
                        f = h5_files[j]
                        new_file_name = base + '_s' + str(start_number + j + 1) + '.h5'
                        shutil.move(os.path.join(self.working_directory, folders_to_process[i], f),
                                    os.path.join(self.working_directory, folders_base[i], new_file_name))

                    # Rename and move txt files from X_r to X folder
                    for j in range(len(txt_files)):
                        f = txt_files[j]
                        new_file_name = base + '_s' + str(start_number + j + 1) + '.txt'
                        shutil.move(os.path.join(self.working_directory, folders_to_process[i], f),
                                    os.path.join(self.working_directory, folders_base[i], new_file_name))

                    # Get rid of temporary folders
                    os.rmdir(os.path.join(self.working_directory, folders_to_process[i]))

        return

    def print_log(self, log_freq=10, print_rel=True):
        if self.solver.iteration % log_freq == 0:
            logger.info('Iter.: %i,\tt: %.3e,\twt: %.1f' % (  # , dt: %.1e
                self.solver.iteration, self.solver.sim_time, self.solver.get_world_time() - self.solver.start_time))
            if print_rel:
                logger.info('Iter.: %.3f%%,\tt: %.3f%%,\twt: %.3f%%\n' % (
                    100 * self.solver.iteration / self.solver.stop_iteration,
                    100 * self.solver.sim_time / self.solver.stop_sim_time,
                    100 * (self.solver.get_world_time() - self.solver.start_time) / self.solver.stop_wall_time))

def myprint(string):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(string)


if __name__ == "__main__":
    Dd = DedalusPy(sys.argv)
    logger.info('Starting loop')
    start_time = time.time()
    try:
        while Dd.solver.proceed:
            u_out = Dd.advance(Dd.t)

        Dd.plot_intermediate()

        plt.figure()
        plt.title('Spectral representation')
        # alternate sign. Reason not clear here...
        signs = np.ones(len(Dd.u['g']))
        signs[1::2] = -1
        coeffs = 1/Dd.nx[0] * scipy.fftpack.dct(Dd.u['g'], type=2) * signs
        coeffs[0] = coeffs[0] / 2
        plt.plot(Dd.u['c'], '+-', markersize=20)
        plt.plot(coeffs, 'x-', markersize=20)
        plt.grid()

        field_c = list(Dd.u['c'])

        i = Dd.nx[0] - 2  # Dd.nx[0] - 2
        j = Dd.nx[0] - 1  # Dd.nx[0] - 1
        del field_c[i]
        del field_c[i]

        f_i_j = Dd.bc_handle.find_fi_fj(i, j, 0, field_c, Dd.xlim[0])
        field_c.insert(i, f_i_j[1])  # i , j
        field_c.insert(i, f_i_j[0])
        field_c = np.array(field_c)
        # print('field c is\n%s\nu[c] is\n%s' % (str(field_c), str(Dd.u['c'])))
        # plt.plot(field_c, 'o')
        plt.legend(['u[\'c\']', 'u cosine transform coefficients', 'reconstruction'])
        plt.show(block=True)

        plt.figure()
        plt.title('Absolute errors')
        plt.semilogy(np.abs(Dd.u['c'] - coeffs), 'x-', markersize=20)
        plt.semilogy(np.abs(Dd.u['c'] - field_c), 'x-', markersize=20)
        plt.grid()
        plt.legend(['Diff. u[\'c\'] and cosine transform', 'Diff. u[\'c\'] and reconst. (i, j) = (%d, %d)' % (i, j)])
        plt.show(block=True)

    except Exception as e:
        logger.error('Exception raised, triggering end of main loop.')
        print(e)
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i' % Dd.solver.iteration)
        logger.info('Sim end time: %f' % Dd.solver.sim_time)
        logger.info('Run time: %.2f sec' % (end_time - start_time))
        logger.info('Run time: %f cpu-hr' % ((end_time - start_time) / 3600 * Dd.domain.dist.comm_cart.size))

