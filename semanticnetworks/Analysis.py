import graph_tool as gt
import numpy as np
import matplotlib.pyplot as plt
import traces

"""
This file contains a collection of tools that are useful for pipelining
the data saved from the concept networks and their nulls, as well as analysis
of that data.
"""

# texts (and ordering we use)
texts = ['Wrong', 'Axler', 'Edwards', 'Lang', 'Petersen', 'Robbiano',
         'Bretscher', 'Greub', 'Hefferson', 'Strang']


def load_n_sents():
    """
    Loads the number of sentences in each text
    """
    n_sents = []
    for text in texts:
        n_sents.append(np.load('Textbooks/{}/n_sents.npy'.format(text)))
    return np.array(n_sents)


def load_index_labels():
    """
    Loads the index sets of the actual networks
    """
    labels = []
    for text in texts:
        labels.append(np.load('Textbooks/{}/index_labels.npy'.format(text)))
    return labels


def load_text():
    """
    Loads the cleaned text of the actual networks
    """
    clean_text = []
    for text in texts:
        clean_text.append(np.load('Textbooks/{}/clean_text.npy'.format(text)))
    return clean_text


def count_occ(word, all_texts, text_lens):
    """
    counts number of occurrences of any word across all texts
    """
    n_occs = []
    for text in all_texts:
        n_occs.append(sum([1 if word in ' '.join(sent) else 0
                           for sent in text]))
    return np.array(n_occs)/text_lens


def avg_occ(word, all_texts, text_lens):
    """
    counts average occurrence position of any word across all texts
    """
    n_occs = []
    for text in all_texts:
        n_occs.append(np.mean([i+1 for i in
                               range(len(text)) if word in ' '.join(text[i])]))
    return np.array(n_occs)/text_lens


def load_sent_coocs():
    """
    Loads the cooccurrence matrices (i.e. weighted graph structures) of
    the actual networks
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/sentence_cooc.npy'.format(text)))
    return mats


def load_sent_filts():
    """
    Loads the sentence-filtration matrices  of the actual networks
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/sentence_filtration.npy'.format(text)))
    return mats


def load_oaat_filt_mats():
    """
    Loads the oaat filtration matrices of the true networks
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/oaat_filt_mats.npy'.format(text)))
    return mats


def load_sent_bars():
    """
    Loads the sentence-barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/sentence_bars.npy'.format(text)))
    return bars


def load_oaat_bars():
    """
    Loads the oaat-barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/oaat_bars.npy'.format(text)))
    return bars


def load_r_ind_cooc_mats():
    """
    Loads the cooccurrence matrices (i.e. weighted graph structures) of the
    random index null models
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/r_ind_cooc_mats.npy'.format(text)))
    return mats


def load_r_ind_filt_mats():
    """
    Loads the filtration matrices of the randomindex null models
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/r_ind_filt_mats.npy'.format(text)))
    return mats


def load_r_ind_oaat_mats():
    """
    Loads the oaat matrices of the randomindex null models
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/r_ind_oaat_mats.npy'.format(text)))
    return mats


def load_r_ind_oaat_bars():
    """
    Loads the random index-barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/r_ind_bars.npy'.format(text)))
    return bars


def load_r_ind_sent_bars():
    """
    Loads the random index-barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/r_ind_sent_bars.npy'.format(text)))
    return bars


def load_r_ind_C():
    """
    Loads the random index core-periphery grouping
    """
    return np.load('Nulls_data/rand_ind_C.npy')


def load_r_ind_qc():
    """
    Loads the random index coreness
    """
    return np.load('Nulls_data/rand_ind_qc.npy')


def load_r_ind_M():
    """
    Loads the random index periphery community grouping
    """
    return np.load('Nulls_data/rand_ind_M.npy')


def load_r_ind_qm():
    """
    Loads the random index periphery modularity
    """
    return np.load('Nulls_data/rand_ind_qm.npy')


def load_r_sentord_C():
    """
    Loads the random index core-periphery grouping
    """
    return np.load('Nulls_data/rand_sentord_C.npy')


def load_r_sentord_qc():
    """
    Loads the random index coreness
    """
    return np.load('Nulls_data/rand_sentord_qc.npy')


def load_r_sentord_M():
    """
    Loads the random index periphery community grouping
    """
    return np.load('Nulls_data/rand_sentord_M.npy')


def load_r_sentord_qm():
    """
    Loads the random index periphery modularity
    """
    return np.load('Nulls_data/rand_sentord_qm.npy')


def load_r_sentord_cooc_mats():
    """
    Loads the cooccurrence matrices (i.e. weighted graph structures) of the
    random index null models
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/r_sentord_cooc_mats.npy'.format(text)))
    return mats


def load_r_sentord_filt_mats():
    """
    Loads the filtration matrices of the randomindex null models
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/r_sentord_filt_mats.npy'.format(text)))
    return mats


def load_r_sentord_oaat_mats():
    """
    Loads the filtration matrices of the randomindex null models
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/r_sentord_oaat_mats.npy'.format(text)))
    return mats


def load_r_sentord_oaat_bars():
    """
    Loads the random index-barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/r_sentord_bars.npy'.format(text)))
    return bars


def load_r_sentord_sent_bars():
    """
    Loads the random index-barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/r_sentord_sent_bars.npy'.format(text)))
    return bars


def load_cont_config_cooc_mats():
    """
    Loads the cooccurrence matrices (i.e. weighted graph structures) of the continuous
    configuration null models
    """
    mats = []
    for text in texts:
        mats.append(np.load('Textbooks/{}/cont_config_mats.npy'.format(text)))
    return mats


def load_cont_config_C():
    """
    Loads the random index core-periphery grouping
    """
    return np.load('Nulls_data/cont_config_C.npy')


def load_cont_config_qc():
    """
    Loads the random index coreness
    """
    return np.load('Nulls_data/cont_config_qc.npy')


def load_cont_config_M():
    """
    Loads the random index periphery community grouping
    """
    return np.load('Nulls_data/cont_config_M.npy')


def load_cont_config_qm():
    """
    Loads the random index periphery modularity
    """
    return np.load('Nulls_data/cont_config_qm.npy')


def load_rand_edge_bars():
    """
    Loads the random edge barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/rand_edge_bars.npy'.format(text)))
    return bars


def load_node_ord_bars():
    """
    Loads the node-ordered barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/node_ord_bars.npy'.format(text)))
    return bars


def load_topo_dist_bars():
    """
    Loads the topological distance barcodes of the actual networks
    """
    bars = []
    for text in texts:
        bars.append(np.load('Textbooks/{}/topo_dist_bars.npy'.format(text)))
    return bars


def group_node_intro(filt, groups, n_sents):
    """
    Looks at the dyanmics of the addition of nodes into the network from the
    core and periphery groups, or other community assignments

    Requires input of the filtration matrix, core-periphery or community group
    assignments, and number of sentences in text
    """
    devs = np.zeros((len(set(groups)), n_sents+1))
    for i in range(len(groups)):
        intro = int(filt[i, i])
        devs[groups[i], intro:] += 1
    return (devs.T / np.max(devs, axis=1)).T


def group_node_intro_times(filt, groups, n_sents):
    """
    Returns lists of addition times of nodes into particular groups
    """
    devs = [[] for _ in range(len(set(groups)))]
    for i in range(len(groups)):
        intro = int(filt[i, i])
        devs[groups[i]].append(intro/n_sents)  # still normalize addition time
    return devs


def group_edge_intro(filt, core_peri, communities, n_sents):
    """
    Looks at the dyanmics of the addition of edges into the network within and
    between various community assignments/core-periphery

    Requires input of the filtration matrix, core-periphery and community group
    assignments, andnumber of sentences in text
    """
    # first, need to transform the core_periphery assignments to replace all
    # periphery info with
    core_peri = 1 - core_peri  # since usually core is 1, peri is 0
    groups = core_peri
    # replace peri with community assignments
    groups[groups == 1] = communities
    # first array will be core -> periphery, second will be inter-periphery,
    # rest will be within-group, starting with core
    devs = np.zeros((len(set(groups))+2, n_sents+1))
    # now go through and count edges at each sentence
    for v in range(1, n_sents+1):
        # gives indices of val
        inds = set([tuple(sorted(x)) for x in zip(*np.where(filt == v))])
        # remove the diagonal terms, we only care about edges
        inds = [ind for ind in inds if ind[0] != ind[1]]

        def ind2ind(ind):
            """
            Helper function to convert double indices to corresonding list inds
            """
            # core-periphery
            if groups[ind[0]] == 0 and groups[ind[1]] != 0:
                return 0
            # inter-periphery
            elif groups[ind[0]] != 0 and groups[ind[0]] != groups[ind[1]]:
                return 1
            elif groups[ind[0]] == groups[ind[1]]:
                return 2 + groups[ind[0]]
        # convert the indices
        new_inds = np.array([ind2ind(ind) for ind in inds], dtype=int)
        # add to dev
        devs[new_inds, v:] += 1
    # normalize and return
    return (devs.T / np.max(devs, axis=1)).T


def normed_diff_area(c1, c2):
    """
Integrates the difference between two "curves" ``c1`` and ``c2``, given as
arrays, assuming the same support (from 0 to 1).
"""
    x = np.linspace(0, 1, len(c1))
    diff = c1 - c2
    return np.trapz(diff, x)


def time_normalize_and_merge(timeseries_lst, make_1_1=False):
    """
    Takes in a list of time series (of possibly different lengths) and
    subsamples them using traces so they all range from 0 and 1 and are defined
    on the same points.
    """
    n_series = len(timeseries_lst)
    timeseries_lst = [traces.TimeSeries(zip(*(np.linspace(0, 1, len(ts)), ts)))
                      for ts in timeseries_lst]
    timeseries_lst = traces.TimeSeries.merge(timeseries_lst)
    if make_1_1:
        timeseries_lst[1.0] = [1.0] * n_series
    X = list(zip(*timeseries_lst.items()))[0]  # x values
    Y = list(zip(*timeseries_lst.items()))[1]  # y values across all series
    return X, Y


def plot_barcode(bars, length, dims=[0, 1, 2], end=True):
    """
    Takes in bars, as generated by, e.g., ripser, and plots the barcode.
    """
    bars = dict(bars)
    count = 1
    has_inf = False
    colors = ['xkcd:emerald green', 'xkcd:tealish', 'xkcd:peacock blue']
    # iterate through dimension
    for d in dims:
        bn = bars[d]
        bn = sorted(bn, key=lambda x: x[0])
        for b, i in zip(bn, range(len(bn))):
            # extend in the case of infinite cycles
            if b[1] == np.inf:
                has_inf = True
                b = (b[0], 1.3*length)
            # plot first one with label
            if i == 0:
                plt.plot(b, [count, count], color=colors[d],
                         label='{}-cycles'.format(d))
            else:
                plt.plot(b, [count, count], color=colors[d])
            count += 1
        count += 1
    # add end of filtration line
    plt.axvline(x=length, color='xkcd:grey', alpha=0.5, linestyle=':')
    if end:
        plt.annotate('Filtration end', (length+10, 0.5*count), rotation=270,
                     color='xkcd:grey', alpha=0.5)
    lims = plt.xlim()
    plt.xlim([-0.05*length, length*1.05])
    plt.xlabel('Sentence number')
    plt.ylabel('Cycle number')


def betti_curves(bars, length):
    """
    Takes in bars and returns the betti curves
    """
    bettis = np.zeros((len(bars), length))
    for i in range(bettis.shape[0]):
        bn = bars[i][1]
        for bar in bn:
            birth = int(bar[0])
            death = length+1 if np.isinf(bar[1]) else int(bar[1]+1)
            bettis[i][birth:death] += 1
    return bettis


def plot_bettis(bettis_lst):
    """
    Takes in a list of bettis (across multiple networks) and plots their
    average plus or minus 2 standard deviations
    """
    colors = ['xkcd:emerald green', 'xkcd:tealish', 'xkcd:peacock blue']
    for i in range(3):
        bettis_i = [betti[i] for betti in bettis_lst]
        X, Y = time_normalize_and_merge(bettis_i)
        mean = np.mean(Y, axis=1)
        mins = np.mean(Y, axis=1) - 2*np.std(Y, axis=1)
        mins = np.maximum(0, mins)
        maxes = np.mean(Y, axis=1) + 2*np.std(Y, axis=1)
        plt.plot(X, mean, color=colors[i], label='$\\beta_{}$'.format(i),
                 linewidth=0.5)
        plt.fill_between(X, mins, maxes, facecolor=colors[i], alpha=0.3)
    plt.xlabel('(Normalized) exposition time')
    plt.ylabel('Number of live cycles')


def barcode_density(bars, length):
    """
    calculates the barcode density (normalized average cycle lifetime)
    of a barcode
    """
    densities = np.zeros(len(bars))
    nums = np.array([len(bars[i][1]) for i in range(len(bars))])
    num_infs = np.zeros(len(bars))
    for i in range(len(bars)):
        tot = 0
        intervals = bars[i][1]
        for intr in intervals:
            if np.isinf(intr[1]):
                num_infs[i] += 1
                tot += (length-intr[0])/(length-1)
            else:
                tot += (intr[1] - intr[0])/(length-1)
        densities[i] = tot
    normed_density = densities/nums
    normed_density[np.isnan(normed_density)] = 0
    return np.stack([densities, nums, normed_density, num_infs])


# Taken from https://gist.github.com/bbengfort/a430d460966d64edc6cad71c502d7005
def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    """
    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')
    """

    """
    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')
    """

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


# adapted from https://gist.github.com/bbengfort/a430d460966d64edc6cad71c502d7005
def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG
