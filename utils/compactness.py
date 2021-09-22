import numpy as np
import tqdm
from scipy.spatial.distance import pdist, squareform

def compress(data, percentile, axis):
    """
    Description:
    Sets all elements of "data" which's norm along the axis "axis" is smaller than the percentile "percentile"

    Args:
    * data (array):  data to be compressed
    * percentile (float): percentile which is used to select the data
    that will be zeroed.
    * axis (int / tuple of ints): which dimension(s) of "data" array
    use to take the norm.

    Returns:
    * data_compressed (array): compressed data

    """
    data_compressed = np.copy(data)
    data_compressed[np.where(
        np.linalg.norm(data_compressed, axis=axis) <
        np.percentile(np.linalg.norm(data_compressed, axis=axis),
                      percentile)), :] = 0
    return data_compressed


def compress_error(data, data_compressed):
    """
    Description:
    Computes the normalized mean squared error of between the original data and the compressed data.

    Args:
    * data (array):  original data
    * data_compressed (array): compressed data

    Returns:
    * compression_error (float) : normalized mean squared error of the compression

    """
    compression_error = np.linalg.norm(data_compressed - data) / \
                        np.linalg.norm(data)
    return compression_error


def compactness_scores(percentiles, ts_gen, sc, sc_surrogate):
    """
    Description:
    Computes the compression performance scores (normalized mean squared error and the correlation) between the original
    data and the compressed data. The scores are computed for all the subjects in the ts_gen loader.
    The scores are computed for compression in the roi space, or in the connectome spectrum, both using the sc
    connectome and the surrogate connectomes in sc_surrogate to compress.

    Args:
    * percentiles (list):  list of percentiles at which the compression is performed
    * ts_gen (utils.data.TsGenerator object): Time series data loader.
    * sc (utils.data.Connectome object): Structural connectivity object.
    * sc_surrogate (list of utils.data.Connectome objects): List of surrogate structural connectivity objects.

    Returns:
    * compression_error_roi: normalized mean squared error of the compression in the roi space
    * correlation_roi:  correlation of the compression in the roi space
    * compression_error_gft: normalized mean squared error of the compression in the connectome spectrum space
    * correlation_gft: correlation of the compression in the connectome spectrum space
    * compression_error_sur_gft: normalized mean squared error of the compression in the surrogate connectome spectrum
    spaces
    * correlation_sur_gft: correlation of the compression in the surrogate connectome spectrum spaces
    """

    n_subjects = ts_gen.nsub
    compression_error_roi = np.zeros((n_subjects, len(percentiles)))
    compression_error_gft = np.zeros((n_subjects, len(percentiles)))

    correlation_roi = np.zeros((n_subjects, len(percentiles)))
    correlation_gft = np.zeros((n_subjects, len(percentiles)))

    compression_error_sur_gft = np.zeros((len(sc_surrogate),
                                          n_subjects,
                                          len(percentiles)))
    correlation_sur_gft = np.zeros((len(sc_surrogate),
                                    n_subjects,
                                    len(percentiles)))

    for s in tqdm.tqdm(range(n_subjects)):
        # load time-series for subject [ROIs x Time x Trials]
        epochs, cond = ts_gen.loader_ts(s)
        # select only FACES trials
        epochs = epochs[:, :, cond == 1]
        # baseline correction
        epochs = epochs - epochs[:, ts_gen.tvec_pre].mean(1, keepdims=True)
        # graph Fourier transform
        epochs_graph = sc.gft(epochs)

        for p, percentile in enumerate(percentiles):
            # compression error and correlation for ROI basis
            compressed_epochs = compress(epochs, percentile, (1, 2))
            compression_error_roi[s, p] = compress_error(epochs.mean(2),
                                                      compressed_epochs.mean(2))
            correlation_roi[s, p] = np.corrcoef(epochs.mean(2).flatten(),
                                                compressed_epochs.mean(2).flatten())[0, 1]

            # compression error for GFT basis
            compressed_epochs_graph = compress(epochs_graph,
                                               percentile, (1, 2))
            compression_error_gft[s, p] = compress_error(epochs_graph.mean(2),
                                                      compressed_epochs_graph.mean(2))
            correlation_gft[s, p] = np.corrcoef(epochs_graph.mean(2).flatten(),
                                                compressed_epochs_graph.mean(2).flatten())[0, 1]

        for r in range(len(sc_surrogate)):
            epochs_rand_graph = sc_surrogate[r].gft(epochs)
            for p, percentile in enumerate(percentiles):
                compressed_epochs_rand_graph = compress(epochs_rand_graph,
                                                        percentile, (1, 2))
                compression_error_sur_gft[r, s, p] = compress_error(epochs_rand_graph.mean(2),
                                                                 compressed_epochs_rand_graph.mean(2))
                correlation_sur_gft[r, s, p] = np.corrcoef(epochs_rand_graph.mean(2).flatten(),
                                                           compressed_epochs_rand_graph.mean(2).flatten())[0, 1]
    return compression_error_roi, correlation_roi, compression_error_gft, correlation_gft, \
           compression_error_sur_gft, correlation_sur_gft


def compactness_scores_geometry_preserving(percentiles, ts_gen, sc_surro_wwp, sc_surro_wsp, sc_surro_wssp,options = ['wwp','wsp','wssp']):
    nsub = ts_gen.nsub

    compression_error_rand_GFT_wwp = np.zeros((len(sc_surro_wwp), nsub,
                                           len(percentiles)))
    correlation_rand_GFT_wwp = np.zeros((len(sc_surro_wwp), nsub, len(percentiles)))

    compression_error_rand_GFT_wsp = np.zeros((len(sc_surro_wsp), nsub,
                                               len(percentiles)))
    correlation_rand_GFT_wsp = np.zeros((len(sc_surro_wsp), nsub, len(percentiles)))

    compression_error_rand_GFT_wssp = np.zeros((len(sc_surro_wssp), nsub,
                                               len(percentiles)))
    correlation_rand_GFT_wssp = np.zeros((len(sc_surro_wssp), nsub, len(percentiles)))

    for s in tqdm.tqdm(range(nsub)):
        # load time-series for subject [ROIs x Time x Trials]
        epochs, cond = ts_gen.loader_ts(s)
        # select only FACES trials
        epochs = epochs[:, :, cond == 1]
        # baseline correction
        epochs = epochs - epochs[:, ts_gen.tvec_pre].mean(1, keepdims=True)

        if 'wwp' in options:
            for r in range(len(sc_surro_wwp)):
                # graph Fourier transform
                epochs_rand_graph_wwp = sc_surro_wwp[r].gft(epochs)
                for p, percentile in enumerate(percentiles):
                    compressed_epochs_rand_graph_wwp = compress(epochs_rand_graph_wwp,
                                                            percentile, (1, 2))
                    compression_error_rand_GFT_wwp[r, s, p] = compress_error(epochs_rand_graph_wwp.mean(2),
                                                                      compressed_epochs_rand_graph_wwp.mean(2))
                    correlation_rand_GFT_wwp[r, s, p] = np.corrcoef(epochs_rand_graph_wwp.mean(2).flatten(),
                                                                compressed_epochs_rand_graph_wwp.mean(2).flatten())[0, 1]
        if 'wsp' in options:
            for r in range(len(sc_surro_wsp)):
                # graph Fourier transform
                epochs_rand_graph_wsp = sc_surro_wsp[r].gft(epochs)
                for p, percentile in enumerate(percentiles):
                    compressed_epochs_rand_graph_wsp = compress(epochs_rand_graph_wsp,
                                                                percentile, (1, 2))
                    compression_error_rand_GFT_wsp[r, s, p] = compress_error(epochs_rand_graph_wsp.mean(2),
                                                                          compressed_epochs_rand_graph_wsp.mean(2))
                    correlation_rand_GFT_wsp[r, s, p] = np.corrcoef(epochs_rand_graph_wsp.mean(2).flatten(),
                                                                    compressed_epochs_rand_graph_wsp.mean(2).flatten())[
                        0, 1]
        if 'wssp' in options:
            for r in range(len(sc_surro_wssp)):
                # graph Fourier transform
                epochs_rand_graph_wssp = sc_surro_wssp[r].gft(epochs)
                for p, percentile in enumerate(percentiles):
                    compressed_epochs_rand_graph_wssp = compress(epochs_rand_graph_wssp,
                                                                percentile, (1, 2))
                    compression_error_rand_GFT_wssp[r, s, p] = compress_error(epochs_rand_graph_wssp.mean(2),
                                                                          compressed_epochs_rand_graph_wssp.mean(2))
                    correlation_rand_GFT_wssp[r, s, p] = np.corrcoef(epochs_rand_graph_wssp.mean(2).flatten(),
                                                                    compressed_epochs_rand_graph_wssp.mean(2).flatten())[
                        0, 1]

    return compression_error_rand_GFT_wwp, correlation_rand_GFT_wwp, compression_error_rand_GFT_wsp, correlation_rand_GFT_wsp, compression_error_rand_GFT_wssp, correlation_rand_GFT_wssp


def compactness_scores_euclidean(percentiles, ts_gen, euc):


    nsub = ts_gen.nsub

    compression_error_GFT_euc = np.zeros((nsub, len(percentiles)))
    correlation_GFT_euc = np.zeros((nsub, len(percentiles)))

    for s in tqdm.tqdm(range(nsub)):
        # load time-series for subject [ROIs x Time x Trials]
        epochs, cond = ts_gen.loader_ts(s)
        # select only FACES trials
        epochs = epochs[:, :, cond == 1]
        # baseline correction
        epochs = epochs - epochs[:, ts_gen.tvec_pre].mean(1, keepdims=True)

        # graph Fourier transform
        epochs_graph_euc = euc[s].gft(epochs)
        for p, percentile in enumerate(percentiles):
            compressed_epochs_graph_euc = compress(epochs_graph_euc,
                                                        percentile, (1, 2))
            compression_error_GFT_euc[ s, p] = compress_error(epochs_graph_euc.mean(2),
                                                                  compressed_epochs_graph_euc.mean(2))
            correlation_GFT_euc[s, p] = np.corrcoef(epochs_graph_euc.mean(2).flatten(),
                                                            compressed_epochs_graph_euc.mean(2).flatten())[
                0, 1]

    return compression_error_GFT_euc, correlation_GFT_euc


def compactness_scores_data_driven(percentiles, ts_gen):
    nsub = ts_gen.nsub
    compression_error_pca = np.zeros((nsub, len(percentiles)))
    correlation_pca = np.zeros((nsub, len(percentiles)))

    compression_error_ica = np.zeros((nsub, len(percentiles)))
    correlation_ica = np.zeros((nsub, len(percentiles)))
    from sklearn.decomposition import PCA, FastICA
    pca = PCA()
    ica = FastICA(max_iter=500)
    erp_mat = []
    for s in tqdm.tqdm(range(nsub)):
        # load time-series for subject [ROIs x Time x Trials]
        epochs, cond = ts_gen.loader_ts(s)
        # select only FACES trials
        epochs = epochs[:, :, cond == 1]
        # baseline correction
        epochs = epochs - epochs[:,ts_gen.tvec_pre].mean(1, keepdims=True)

        erp_mat.append(epochs.mean(2))

    erp_mat = np.transpose(np.array(erp_mat), axes=(1, 2, 0))
    pca.fit(epochs.reshape(erp_mat.shape[0], -1).T)
    ica.fit(epochs.reshape(erp_mat.shape[0], -1).T)

    for s in tqdm.tqdm(range(nsub)):
        # load time-series for subject [ROIs x Time x Trials]
        epochs, cond = ts_gen.loader_ts(s)
        # select only FACES trials
        epochs = epochs[:, :, cond == 1]
        # baseline correction
        epochs = epochs - epochs[:, ts_gen.tvec_pre].mean(1, keepdims=True)

        # transforms
        epochs_pca = pca.transform(epochs.reshape(epochs.shape[0], -1).T).T.reshape(*epochs.shape)
        epochs_ica = ica.transform(epochs.reshape(epochs.shape[0], -1).T).T.reshape(*epochs.shape)

        for p, percentile in enumerate(percentiles):
            # compression error and correlation for PCA basis
            compressed_epochs = compress(epochs_pca, percentile, axis=(1, 2))
            compression_error_pca[s, p] = compress_error(epochs_pca.mean(2),
                                                      compressed_epochs.mean(2))
            correlation_pca[s, p] = np.corrcoef(epochs_pca.mean(2).flatten(),
                                                compressed_epochs.mean(2).flatten())[0, 1]

            # compression error and correlation for ICA basis
            compressed_epochs = compress(epochs_ica, percentile, axis=(1, 2))
            compression_error_ica[s, p] = compress_error(epochs_ica.mean(2),
                                                      compressed_epochs.mean(2))
            correlation_ica[s, p] = np.corrcoef(epochs_ica.mean(2).flatten(),
                                                compressed_epochs.mean(2).flatten())[0, 1]
    return compression_error_pca, correlation_pca, compression_error_ica, correlation_ica


def compactness_dynamics(ts_gen, sc, sc_surrogate, percentiles):
    """
    Description:
    Computes the compression error (normalized mean squared error and the correlation) between the original
    data and the compressed data at each time-point. The scores are computed for all the subjects in the ts_gen loader.
    The scores are computed for compression in the roi space, or in the connectome spectrum, both using the sc
    connectome and the surrogate connectomes in sc_surrogate to compress.

    Args:
    * ts_gen (utils.data.TsGenerator object): Time series data loader.
    * sc (utils.data.Connectome object): Structural connectivity object.
    * sc_surrogate (list of utils.data.Connectome objects): List of surrogate structural connectivity objects.
    * percentiles (list):  list of percentiles at which the compression is performed

    Returns:
    * compression_time_error_roi: normalized mean squared error of the compression in the roi space for each time point
    * compression_time_error_gft: normalized mean squared error of the compression in the connectome spectrum space for
    each time point
    """

    nsub = ts_gen.nsub
    tvec_analysis = ts_gen.tvec_analysis
    compression_time_error_roi = np.zeros((nsub, len(tvec_analysis)))
    compression_time_error_gft = np.zeros((nsub, len(tvec_analysis)))

    for s in tqdm.tqdm(range(nsub)):
        # load time-series for subject [ROIs x Time x Trials]
        epochs, cond = ts_gen.loader_ts(s)
        # select only FACES trials
        epochs = epochs[:, :, cond == 1]
        # baseline correction
        epochs = epochs - epochs[:, ts_gen.tvec_pre].mean(1, keepdims=True)
        # graph Fourier transform
        epochs_graph = sc.gft(epochs)

        for t in range(len(tvec_analysis)):
            compression_roi = np.zeros(len(percentiles))
            compression_gft = np.zeros(len(percentiles))

            for p, percentile in enumerate(percentiles):
                compressed_epochs = compress(epochs[:, t], percentile, 1)
                compression_roi[p] = compress_error(epochs[:, t].mean(1),
                                                 compressed_epochs.mean(1))

                compressed_epochs = compress(epochs_graph[:, t], percentile, 1)
                compression_gft[p] = compress_error(epochs_graph[:, t].mean(1),
                                                 compressed_epochs.mean(1))

            compression_time_error_roi[s, t] = np.mean(compression_roi)
            compression_time_error_gft[s, t] = np.mean(compression_gft)

    return compression_time_error_roi, compression_time_error_gft
