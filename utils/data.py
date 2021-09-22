import os
import numpy as np
import pandas as pd
import pygsp
import scipy
import scipy.io
import os

class Connectome(pygsp.graphs.Graph):
    def __init__(self,sc_file):
        super().__init__(sc_file, lap_type='normalized')
        self.compute_fourier_basis()

def loader_sc(scale, datadir, sc_type='num'):
    # Load ROI atlas info
    roifname = os.path.join(datadir, 'Lausanne2008_Yeo7RSNs.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]

    # Load ROI structural connectivity (SC) info
    if sc_type == 'len':
        sc_fname = os.path.join(datadir, 'SC_betzel',
                                'SC_len_betzel_scale_{}.mat'.format(scale))
        sc_file = scipy.io.loadmat(sc_fname)['dist'][cort][:, cort]
        sc_file[sc_file != 0] = 1 / ((sc_file[sc_file != 0]) / np.max(sc_file))
    elif sc_type == 'num':
        sc_fname = os.path.join(datadir, 'SC_betzel',
                                'SC_num_betzel_scale_{}.mat'.format(scale))
        sc_file = scipy.io.loadmat(sc_fname)['num'][cort][:, cort]
        sc_file[sc_file != 0] = np.log(sc_file[sc_file != 0])

    sc_file[np.isnan(sc_file)] = 0
    sc_file[np.isinf(sc_file)] = 0
    sc = Connectome(sc_file)
    return sc


def loader_sc_surrogates(sc_rand_dir):
    sc_fname_list = os.listdir(sc_rand_dir)
    rand_scs = []
    for sc_fname in sc_fname_list:
        sc_file = np.load(os.path.join(sc_rand_dir, sc_fname)) # These are already log normalized and diag-zeroed

        sc = Connectome(sc_file)
        rand_scs.append(sc)

    return rand_scs

def loader_sc_surrogates_geometry_preserving(sc_rand_dir,scale,datadir):
    #sc_fname_list = os.listdir(sc_rand_dir)
    roifname = os.path.join(datadir, 'Lausanne2008_Yeo7RSNs.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]

    sc_fname_list = [elem for elem in os.listdir(sc_rand_dir) if elem.startswith('rand_SC_')]
    wwp_fname_list = [elem for elem in sc_fname_list if elem.endswith('Wwp.mat')]
    wsp_fname_list = [elem for elem in sc_fname_list if elem.endswith('Wsp.mat')]
    wssp_fname_list = [elem for elem in sc_fname_list if elem.endswith('Wssp.mat')]

    rand_sc_wwp = []
    rand_sc_wsp = []
    rand_sc_wssp = []

    for sc_fname in wwp_fname_list:
        sc_file = scipy.io.loadmat(os.path.join(sc_rand_dir, sc_fname))['Wwp']
        sc_file = sc_file[cort][:, cort]
        sc_file[sc_file != 0] = np.log(sc_file[sc_file != 0])
        sc_file[np.isnan(sc_file)] = 0
        sc_file[np.isinf(sc_file)] = 0

        sc = pygsp.graphs.Graph(sc_file, lap_type='normalized')
        sc.compute_fourier_basis()
        rand_sc_wwp.append(sc)

    for sc_fname in wsp_fname_list:
        sc_file = scipy.io.loadmat(os.path.join(sc_rand_dir, sc_fname))['Wsp']
        sc_file = sc_file[cort][:, cort]
        sc_file[sc_file != 0] = np.log(sc_file[sc_file != 0])
        sc_file[np.isnan(sc_file)] = 0
        sc_file[np.isinf(sc_file)] = 0

        sc = pygsp.graphs.Graph(sc_file, lap_type='normalized')
        sc.compute_fourier_basis()
        rand_sc_wsp.append(sc)

    for sc_fname in wssp_fname_list:
        sc_file = scipy.io.loadmat(os.path.join(sc_rand_dir, sc_fname))['Wssp']
        sc_file = sc_file[cort][:, cort]
        sc_file[sc_file != 0] = np.log(sc_file[sc_file != 0])
        sc_file[np.isnan(sc_file)] = 0
        sc_file[np.isinf(sc_file)] = 0

        sc = pygsp.graphs.Graph(sc_file, lap_type='normalized')
        sc.compute_fourier_basis()
        rand_sc_wssp.append(sc)

    return rand_sc_wwp, rand_sc_wsp, rand_sc_wssp

def loader_sc_non_norm(scale, datadir, sc_type='num'):
    # Load ROI atlas info
    roifname = os.path.join(datadir, 'Lausanne2008_Yeo7RSNs.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]

    # Load ROI structural connectivity (SC) info
    if sc_type == 'len':
        sc_fname = os.path.join(datadir, 'SC_betzel',
                                'SC_len_betzel_scale_{}.mat'.format(scale))
        sc_file = scipy.io.loadmat(sc_fname)['dist'][cort][:, cort]
    elif sc_type == 'num':
        sc_fname = os.path.join(datadir, 'SC_betzel',
                                'SC_num_betzel_scale_{}.mat'.format(scale))
        sc_file = scipy.io.loadmat(sc_fname)['num'][cort][:, cort]

    sc = Connectome(sc_file)
    return sc


def loader_sc_surrogates_non_norm(sc_rand_dir):
    sc_fname_list = os.listdir(sc_rand_dir)
    rand_scs = []
    for sc_fname in sc_fname_list:
        sc_file = np.load(os.path.join(sc_rand_dir, sc_fname)) # These are already log normalized and diag-zeroed

        sc = Connectome(sc_file)
        rand_scs.append(sc)

    return rand_scs


def loader_sc_surrogates_geometry_preserving_non_norm(sc_rand_dir,scale,datadir):
    # sc_fname_list = os.listdir(sc_rand_dir)
    roifname = os.path.join(datadir, 'Lausanne2008_Yeo7RSNs.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]

    sc_fname_list = [elem for elem in os.listdir(sc_rand_dir) if elem.startswith('rand_SC_')]
    wwp_fname_list = [elem for elem in sc_fname_list if elem.endswith('Wwp.mat')]
    wsp_fname_list = [elem for elem in sc_fname_list if elem.endswith('Wsp.mat')]
    wssp_fname_list = [elem for elem in sc_fname_list if elem.endswith('Wssp.mat')]

    rand_sc_wwp = []
    rand_sc_wsp = []
    rand_sc_wssp = []

    for sc_fname in wwp_fname_list:
        sc_file = scipy.io.loadmat(os.path.join(sc_rand_dir, sc_fname))['Wwp']
        sc_file = sc_file[cort][:, cort]

        sc = pygsp.graphs.Graph(sc_file, lap_type='normalized')
        sc.compute_fourier_basis()
        rand_sc_wwp.append(sc)

    for sc_fname in wsp_fname_list:
        sc_file = scipy.io.loadmat(os.path.join(sc_rand_dir, sc_fname))['Wsp']
        sc_file = sc_file[cort][:, cort]

        sc = pygsp.graphs.Graph(sc_file, lap_type='normalized')
        sc.compute_fourier_basis()
        rand_sc_wsp.append(sc)

    for sc_fname in wssp_fname_list:
        sc_file = scipy.io.loadmat(os.path.join(sc_rand_dir, sc_fname))['Wssp']
        sc_file = sc_file[cort][:, cort]

        sc = pygsp.graphs.Graph(sc_file, lap_type='normalized')
        sc.compute_fourier_basis()
        rand_sc_wssp.append(sc)

    return rand_sc_wwp, rand_sc_wsp, rand_sc_wssp


class TsGenerator:
    def __init__(self, subject_list, scale, dataset, datadir, tvec_analysis,
                 tvec_pre):
        self.subject_list = subject_list
        self.scale = scale
        self.datadir = datadir
        self.dataset = dataset
        self.tvec_analysis = tvec_analysis
        self.tvec_pre = tvec_pre
        self.nsub = len(subject_list)

    def loader_ts(self, subject):
        tcs_1d = np.load(os.path.join(self.datadir, 'sourcedata', self.dataset,
                                      'scale{}/sub-{}.npy'.format(self.scale,
                                                                  str(self.subject_list[subject]).zfill(2))))
        if self.dataset == 'Faces':
            behav_fname = os.path.join(self.datadir, 'derivatives', 'eeglab',
                                       'sub-'+str(self.subject_list[subject]).zfill(2),
                                       'sub-'+str(self.subject_list[subject]).zfill(2) +
                                       '_FACES_250HZ_behav.npy')
            cond = np.load(behav_fname)
            cond = np.array(list(map(int, (cond == 'FACES'))))

        elif self.dataset == 'Motion':
            behav_fname = os.path.join(self.datadir, 'derivatives', 'eeglab',
                                       'sub-'+str(self.subject_list[subject]).zfill(2),
                                       'sub-'+str(self.subject_list[subject]).zfill(2) +
                                       '_MOTION_250HZ_behav.npy')
            cond = np.load(behav_fname)

        return tcs_1d[:, self.tvec_analysis] / np.std(tcs_1d.mean(2)), cond
