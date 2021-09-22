import numpy as np
import tqdm
from utils.data import TsGenerator


def smoothness(U, G):
    if len(U.shape) > 1:
        # U is [dims x trials]
        dims, trials = U.shape
        s = np.zeros(trials)
        for trl in range(trials):
            u = U[:, trl].reshape(dims, 1)
            s[trl] = (u.T @ G.L @ u) / (u.T @ u)
        s = s.mean()

    else:
        # U is [dims]
        u = U.reshape(-1, 1)
        s = (u.T @ G.L @ u) / (u.T @ u)

    return s


def sdi(sc, ts_generator):
    nsub = ts_generator.nsub
    tvec_analysis = ts_generator.tvec_analysis
    tvec_pre = ts_generator.tvec_pre
    power = np.zeros((sc.N, len(tvec_analysis), nsub))
    evoked_hat = np.zeros((sc.N, len(tvec_analysis), nsub))

    for s in tqdm.tqdm(range(nsub)):
        # load time-series for subject [ROIs x Time x Trials]
        epochs, cond = ts_generator.loader_ts(s)
        # select only FACES trials
        epochs = epochs[:, :, cond == 1]
        # baseline correction
        epochs = epochs - epochs[:, tvec_pre].mean(1, keepdims=True)
        # evoked
        epochs = epochs.mean(2)
        # graph Fourier transform
        epochs_hat = sc.gft(epochs)
        epochs_hat /= np.std(epochs_hat.flatten()) - \
            np.mean(epochs_hat.flatten())
        evoked_hat[:, :, s] = epochs_hat.copy()

        # power of the signal in the graph
        power[:, :, s] = (epochs_hat ** 2)

    # mean across time
    psd = np.mean(power, 1)

    # mean across subjects
    m_psd = np.mean(psd, 1)

    # area under the curve
    auc_tot = np.trapz(m_psd)
    i = 0
    auc = 0
    while auc < (auc_tot / 2):
        i += 1
        auc = np.trapz(m_psd[:i])
    # cut-off frequency
    nn = i - 1
    vhigh = np.zeros_like(sc.U)
    vlow = np.zeros_like(sc.U)
    vhigh[:, nn:] = sc.U[:, nn:]
    vlow[:, :nn] = sc.U[:, :nn]

    # coupling / decoupling

    x_c = np.zeros((sc.N, len(tvec_analysis), nsub))
    x_d = np.zeros((sc.N, len(tvec_analysis), nsub))
    n_c = np.zeros((sc.N, nsub))
    n_d = np.zeros((sc.N, nsub))

    for s in tqdm.tqdm(range(nsub)):
        x_c[:, :, s] = vlow @ evoked_hat[:, :, s]
        x_d[:, :, s] = vhigh @ evoked_hat[:, :, s]

    return psd, nn, x_d, x_c
