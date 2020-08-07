from nltk import word_tokenize, sent_tokenize
import networkx as nx
import enchant
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from itertools import *
from tqdm import tqdm_notebook
import re
import RAKE
from ripser import ripser
import spacy
from nltk.corpus import brown

nlp = spacy.load('en', disable=['parser', 'ner'])

"""
Load the list of non-mathematical words from the Brown corpus;
if not saved, generate this list.
"""
try:
    bwords = str(np.load('bwords.npy'))
except:
    cats = brown.categories()
    cats.remove('learned')
    nlp.max_length = 10000000

    brn = ' '.join(brown.words(categories=cats))
    doc = nlp(brn)
    bwords = ' '.join([w.lemma_.lower() for w in doc])
    np.save('bwords.npy', bwords)

d = enchant.Dict("en_US")  # For spell-checking
# List of stop words
stop_words = [x for x in RAKE.RanksNLLongStopList()
              if len(x) > 1 or x == 'a']
stop_words.remove('value')
vowels = {'a', 'e', 'i', 'o', 'u', 'y'}


def freq_wholelabel(ind_lst, txt):
    """
    Counts frequencies of the words contained in ``ind_lst``
    in the text ``txt``
    """
    freqs = []
    for ind in ind_lst:
        freqs.append(txt.count(ind))
    return {k: f for k, f in zip(ind_lst, freqs)}


class Concept_Network:
    """
    Class wrapping semantic networks: enables the construction of semantic
    networks from text.
    """

    def __init__(self):
        self.index = None
        self.text = None  # text should not change
        self.windows = None  # windows is sliding windows based on text
        self.dist_mat = None
        self.cooc_mat = None
        self.graph = None
        self.index_labels = None
        self.timeline = None
        self.cutoff = None
        self.bars = []
        self.cooc_mat = None
        self.weight_dist = None

    def clean(self, textname, n_chapt):
        """
        Cleans a text by the name of ``textname``, which has ``n_chapt``
        chapters.

        Assumes that texts are stored in a Textbooks directory, and each text
        has an associated folder containing each separate text chapter as a
        .txt file.
        """
        alltext = []
        for i in range(1, n_chapt+1):
            # Read text
            with open('Textbooks/{0}/chapter_{1}.txt'.format(textname,
                                                             i)) as chapter:
                text = chapter.read()
            import unicodedata
            # Text normalization/cleaning
            text = unicodedata.normalize('NFKD',
                                         text).encode('ascii', 'ignore')
            text = b' '.join(text.split()).decode()
            text = text.replace('-', ' ')
            text = nlp(text)
            # Lemmatization
            text = ' '.join(w.lemma_ for w in text)
            sents = sent_tokenize(text)
            sents = [word_tokenize(s) for s in sents]
            # replace all strings with numbers with #
            sents = [[w if not re.search(r"\d+", w) else '#' for w in s]
                     for s in sents]
            # remove non-alphanumeric
            sents = [[w for w in s if re.search('[^a-zA-Z-#]+', w) is None]
                     for s in sents]
            sents = [[w.lower() for w in s] for s in sents]
            text = sents

            def quick_check(word):
                """
                Helper function to screen out non-words to cast as variables.
                Ensures word contains a vowel, spell checks, etc.
                """
                if word == '#':
                    return True
                elif not vowels.intersection(word):
                    return False
                elif len(word) <= 2 and word not in stop_words:
                    return False
                elif 2 < len(word) <= 4 and not d.check(word):
                    return False
                else:
                    return True
            # replace all potential variables with VAR
            text = [[w if quick_check(w) else 'VAR' for w in s] for s in text]
            alltext += text
        self.text = list(filter(lambda x: len(x) > 0, alltext))
        self.windows = self.text
        return self.text

    def capital_lemma(self, spacy_word):
        """
        Helper function that returns the lemmatized version of a word via
        spacy, but maintains the same capitalization as the original.
        """
        out_word = ""
        if spacy_word.shape_[0] == "X":
            out_word += spacy_word.lemma_[0].upper()
        else:
            out_word += spacy_word.lemma_[0]
        out_word += spacy_word.lemma_[1:]
        return out_word

    def clean_notlower(self, textname, n_chapt):
        """
        Auxiliary text cleaning function (similar to ``self.clean``) that does
        not uniformly make words lower-cased; maintains same capitalization as
        original text.
        """
        alltext = []
        for i in range(1, n_chapt+1):
            with open('Textbooks/{0}/chapter_{1}.txt'.format(textname,
                                                             i)) as chapter:
                text = chapter.read()
            import unicodedata
            text = unicodedata.normalize('NFKD',
                                         text).encode('ascii', 'ignore')
            text = b' '.join(text.split()).decode()
            text = text.replace('-', ' ')
            text = nlp(text)
            text = ' '.join(self.capital_lemma(w) for w in text)
            sents = sent_tokenize(text)
            sents = [word_tokenize(s) for s in sents]
            # replace all things with numbers with #
            sents = [[w if not re.search(r"\d+", w) else '#' for w in s]
                     for s in sents]
            # remove non-alphanumeric
            sents = [[w for w in s if re.search('[^a-zA-Z-#]+', w) is None]
                     for s in sents]
            text = sents

            def quick_check(word):
                """
                Helper function to screen out non-words to cast as variables.
                Ensures word contains a vowel, spell checks, etc.
                """
                if word == '#':
                    return True
                elif not vowels.intersection(word):
                    return False
                elif len(word) <= 2 and word not in stop_words:
                    return False
                elif 2 < len(word) <= 4 and not d.check(word):
                    return False
                else:
                    return True
            # replace all potential variables with VAR
            text = [[w if quick_check(w) else 'VAR' for w in s] for s in text]
            alltext += text
        self.text = list(filter(lambda x: len(x) > 0, alltext))
        self.windows = self.text
        return self.text

    def RAKE_keywords(self):
        """
        Extracts the RAKE keywords from the text
        """
        # Additional, non-mathematically-contentful stop-words to consider
        # in the RAKE run
        extra_sl = ['example', 'counterexample', 'text', 'texts',
                    'undergraduate', 'chapter', 'definition',
                    'notation', 'proof', 'exercise', 'result']
        sl = nlp(' '.join(stop_words + extra_sl))
        sl = [w.lemma_ for w in sl]
        rake = RAKE.Rake(sl + ['VAR', '#', '-pron-', '-pron'])
        text = '. '.join([' '.join(x) for x in self.text])
        # Run RAKE
        kws = rake.run(text, minCharacters=3, maxWords=4, minFrequency=5)
        keyphrases = [' '.join([j for j in k[0].split(' ')
                      if j not in {'var', '#', '-pron-', '-pron'}])
                      for k in kws]
        keyphrases = set([k for k in keyphrases if len(k) > 0])
        index = list(map(lambda x: x.split(), keyphrases))
        self.index = index
        return index

    # determines order of first occurrence of index terms.
    # prunes out the index elements which do not occur within the text
    # (i.e., within a single sentence), and orders them w.r.t. first occurrence
    # very very important to run before doing anything else
    def occurrence(self, debug=False, n_allowed=None):
        """
        Does two things:
        (1) Ensures that all the extracted index terms actually appear within
        the text (so we don't have any extracted nodes with no data)
        (2) Goes through the text to determine the order of introduction of
        concepts, so that the resulting co-occurence matrices have indices
        ordered by introduction.
        """
        sentences = self.text
        index = list(map(set, self.index))
        index_labels = np.array(list(map(lambda i: ' '.join(i), self.index)))
        first_occs = []  # sentence number of first term occurrence
        counts = []
        conn_text = '. '.join([' '.join(s) for s in sentences])
        for ind in index_labels:
            for sent, n in zip(sentences, range(1, len(sentences)+1)):
                if ind in ' '.join(sent):
                    first_occs.append(n)
                    break
            else:
                first_occs.append(0)
            # count how many times index term appears in the text
            counts.append(conn_text.count(ind))
        counts = sorted(list(zip(index_labels, counts)),
                        key=lambda x: x[1], reverse=True)
        ordered = [c[0] for c in counts]
        first_occs = np.array(first_occs)
        nonzeros = np.nonzero(first_occs)[0]
        zeros = np.where(first_occs == 0)[0]
        # Yield of how many extracted concepts were actually found
        print("Yield percent:", len(nonzeros)/len(first_occs))
        if debug:
            print(index_labels[zeros])
        # Remove terms which do not occur
        index = np.array(index)[nonzeros]
        index_labels = index_labels[nonzeros]
        # Sort remaining into order of introduction
        sort = np.argsort(first_occs[nonzeros])
        index = index[sort]
        index_labels = index_labels[sort]
        index_labels = np.array([x for x in index_labels
                                 if x in ordered[:n_allowed]])
        index = np.array(list(map(lambda i: set(i.split()), index_labels)))
        self.first_occs = np.sort(first_occs[nonzeros])
        self.index_labels = index_labels
        self.index = index

    def sliding_window(self, lst, size):
        """
        Generates a list of text sliding windows
        of size ``size`` from a sentence, represented as a list of words
        ``lst``. Unused in analysis, but could be useful if you are interested
        in co-occurence on a smaller scale than sentence-level.
        """
        windows = []
        for i in range(0, max(1, len(lst)-size+1)):
            b, e = i, min(i+size, len(lst))
            windows.append(lst[b:e])
        return windows

    def sliding_window_constr(self, size):
        """
        Gets all sliding windows of size ``size`` from the text.
        """
        windows = []
        if self.text is not None:
            for sent in self.text:
                windows += self.sliding_window(sent, size)
        self.windows = windows

    def cooc_2d(self):
        """
        Calculates the number of co-occurrences of the index phrases in the
        text (output as the matrix ``cooc_mat``), as well as records the first
        time a co-occurrence occurred in the text (``dist_mat``).
        """
        dim = len(self.index_labels)
        cooc_mat = np.zeros((dim, dim))
        first_cooc = np.zeros((dim, dim))
        sentences = self.windows
        timeline = {}
        for sent, num in tqdm_notebook(list(zip(sentences, range(1, len(sentences)+1)))):
            joined_sent = ' '.join(sent)
            for i in range(dim):
                if self.index_labels[i] in joined_sent:
                    cooc_mat[i, i] += 1
                    if first_cooc[i, i] == 0:
                        first_cooc[i, i] = num
                    for j in range(i+1, dim):
                        if self.index_labels[j] in joined_sent:
                            cooc_mat[(np.array([i, j]), np.array([j, i]))] += 1
                            if first_cooc[i, j] == 0:
                                first_cooc[(np.array([i, j]), np.array([j, i]))] = num
                                timeline[tuple(self.index_labels[np.array([i, j])])] = num
        first_cooc[first_cooc == 0] = np.inf
        self.cutoff = len(sentences)
        self.dist_mat = first_cooc
        self.cooc_mat = cooc_mat
        self.timeline = timeline
        # make graph
        G = nx.from_numpy_array(cooc_mat)
        G.remove_edges_from(nx.selfloop_edges(G))
        name_mapping = {i: label for i, label in enumerate(self.index_labels)}
        nx.relabel_nodes(G, name_mapping, copy=False)
        self.graph = G
        return cooc_mat, first_cooc, timeline

    def cooc_for_text(self, text_name, nchapts,
                      n_allowed=None, groups=None):
        """
        Wrapper function that effectively does everything - clean text,
        extract the RAKE keyphrases, calculate occurrence within the texts,
        rank the keyphrases based off the RAKE score and inverse Brown
        frequency, and calculate co-occurrence throughout the text.
        """
        # Clean the text
        self.clean(text_name, nchapts)
        # Extract RAKE keyphrases
        self.RAKE_keywords()
        # Calculate occurrence of the keyphrases
        self.occurrence()
        brown_whole = freq_wholelabel(self.index_labels, bwords)
        # Do augmented rake-IDF scoring
        # first re-run rake, just for convenience
        extra_sl = ['example', 'counterexample', 'text', 'texts',
                    'undergraduate', 'chapter', 'definition', 'notation',
                    'proof', 'exercise', 'result']
        sl = nlp(' '.join(stop_words + extra_sl))
        sl = [w.lemma_ for w in sl]
        rake = RAKE.Rake(sl + ['VAR', '#', '-pron-'])
        text = '. '.join([' '.join(x) for x in self.text])
        kws = rake.run(text, minCharacters=3, maxWords=4, minFrequency=5)
        keyphrases = [(' '.join([j for j in k[0].split(' ')
                                 if j not in {'var', '#', '-pron-', '-pron'}]),
                       k[1]) for k in kws]
        keyphrases = [k for k in keyphrases if len(k[0]) > 0]
        rake_dict = {}
        # give each the maximal associated score
        for k, v in keyphrases:
            if k not in rake_dict.keys() or rake_dict[k] < v:
                rake_dict[k] = v
        # now calculate scores
        scores = []
        for ind in self.index_labels:
            score = rake_dict[ind]
            # Add one to prevent divide by zero errors
            brownscore = brown_whole[ind] + 1
            score /= brownscore
            scores.append((ind, score))
        # take the top half of all scored keyphrases
        phrases = sorted(scores, key=lambda x: x[1], reverse=True)
        phrases = phrases[:int(len(phrases)/2)]
        index = list(map(lambda x: x[0].split(), phrases))
        self.index = index
        self.occurrence()
        # then calculate cooccurence
        t = self.cooc_2d()
        return t

    def cooc_for_text_notlower(self, text_name, nchapts,
                               n_allowed=None, groups=None):
        """
        Same thing as ``self.cooc_for_text``, but without forcing lower case.
        """
        self.clean_notlower(text_name, nchapts)
        self.RAKE_keywords()
        self.occurrence()
        brown_whole = freq_wholelabel(self.index_labels, bwords)
        # Do augmented rake-IDF scoring
        # first re-run rake, just for convenience
        extra_sl = ['example', 'counterexample', 'text', 'texts',
                    'undergraduate', 'chapter', 'definition', 'notation',
                    'proof', 'exercise', 'result']
        sl = nlp(' '.join(stop_words + extra_sl))
        sl = [w.lemma_ for w in sl]
        rake = RAKE.Rake(sl + ['VAR', '#', '-pron-'])
        text = '. '.join([' '.join(x) for x in self.text])
        kws = rake.run(text, minCharacters=3, maxWords=4, minFrequency=5)
        keyphrases = [(' '.join([j for j in k[0].split(' ')
                                 if j not in {'var', '#', '-pron-', '-pron'}]),
                       k[1]) for k in kws]
        keyphrases = [k for k in keyphrases if len(k[0]) > 0]
        rake_dict = {}
        # give each the maximal associated score
        for k, v in keyphrases:
            if k not in rake_dict.keys() or rake_dict[k] < v:
                rake_dict[k] = v
        # now calculate scores
        scores = []
        for ind in self.index_labels:
            score = rake_dict[ind]
            brownscore = brown_whole[ind] + 1
            score /= brownscore
            scores.append((ind, score))
        # take the top half
        phrases = sorted(scores, key=lambda x: x[1], reverse=True)
        phrases = phrases[:int(len(phrases)/2)]
        index = list(map(lambda x: x[0].split(), phrases))
        self.index = index
        self.occurrence()
        # then calculate cooccurence
        t = self.cooc_2d()
        return t

    def cont_config_graph(self):
        """
        Generates a continuous configuration null graph with same degree
        distribution/strength distribution as the extracted semantic network.
        See paper for more details.

        Can only be run after runnin ``self.cooc_for_text``.
        """
        nodes = list(self.graph.nodes)
        degs = dict(self.graph.degree())
        dT = sum(degs.values())
        strengths = {n: sum([x[2]['weight']
                             for x in self.graph.edges(n, data=True)])
                     for n in nodes}
        sT = sum(strengths.values())
        # see if we've already computed the best-fit of the normed weights
        if self.weight_dist is not None:
            dist = self.weight_dist
        else:
            # calculate normed weights to determine distribution parameters
            normedweights = []
            for x in self.graph.edges(data=True):
                s0, s1 = strengths[x[0]], strengths[x[1]]
                d0, d1 = degs[x[0]], degs[x[1]]
                s_uv = s0*s1/sT
                d_uv = d0*d1/dT
                normedweights.append(x[2]['weight']*d_uv/s_uv)
            # Candidate distributions
            DISTRIBUTIONS = [st.pareto, st.lognorm, st.levy, st.dweibull,
                             st.burr, st.fisk, st.loggamma, st.loglaplace,
                             st.powerlaw]
            results = []
            normedweights = np.array(normedweights)
            # find the best fit
            for dist in DISTRIBUTIONS:
                try:
                    # attempt fit
                    pars = dist.fit(normedweights)
                    mle = dist.nnlf(pars, normedweights)
                    results.append((mle, dist.name, pars))
                except:
                    pass
            best_fit = sorted(results, key=lambda d: d[0])
            print(best_fit[0])
            dist = getattr(st, best_fit[0][1])(*best_fit[0][2])
            self.weight_dist = dist
        # construct the null graph
        null_graph = np.zeros((len(nodes), len(nodes)))
        for i in range(0, len(nodes)):
            for j in range(i+1, len(nodes)):
                d_uv = degs[nodes[i]]*degs[nodes[j]]/dT
                if np.random.rand() < d_uv:
                    s_uv = strengths[nodes[i]]*strengths[nodes[j]]/sT
                    xi = self.weight_dist.rvs()
                    null_graph[i, j] = xi*s_uv/d_uv
                    null_graph[j, i] = xi*s_uv/d_uv
        return null_graph

    def random_index_null(self, return_index=False, return_otherstuff=False):
        """
        Generates a random index null filtration for the text with a set of
        randomly-selected words from the body of the text. See paper for more
        details.
        """
        tmp_index = self.index.copy()
        tmp_index_labels = self.index_labels.copy()
        tmp_cutoff = self.cutoff
        tmp_first_cooc = self.dist_mat.copy()
        tmp_cooc_tensor = self.cooc_mat.copy()
        tmp_timeline = self.timeline.copy()
        tmp_G = self.graph.copy()
        tmp_first_occs = self.first_occs.copy()
        extra_sl = ['example', 'counterexample', 'text', 'texts', 'undergraduate', 'chapter', 'definition', 'notation',
                    'proof', 'exercise', 'result']
        sl = nlp(' '.join(stop_words + extra_sl))
        sl = [w.lemma_ for w in sl]
        textwds = set(sum(self.text, []))
        textwds.difference_update(sl + ['VAR', '#', '-pron-', '-pron']) #
        random_index = np.random.choice(list(textwds),
                                        size=len(tmp_index),
                                        replace=False)
        self.index_labels = random_index.copy()
        self.index = list(map(lambda x: x.split(), self.index_labels))
        self.occurrence()
        self.windows = self.text
        t = self.cooc_2d()
        new_first_occs = self.first_occs.copy()
        new_cutoff = self.cutoff
        self.index = tmp_index
        self.index_labels = tmp_index_labels
        self.first_occs = tmp_first_occs
        self.graph = tmp_G
        self.cutoff = tmp_cutoff
        self.dist_mat = tmp_first_cooc
        self.cooc_mat = tmp_cooc_tensor
        self.timeline = tmp_timeline
        if return_index:
            return t, random_index
        if return_otherstuff:
            return t, new_first_occs, new_cutoff, random_index
        else:
            return t

    def rnd_sent_ord_null(self):
        """
        Generates a null filtration matrix for the text with the same index,
        but with sentences randomly shuffled. See paper for more details.
        """
        tmp_index = self.index.copy()
        tmp_index_labels = self.index_labels.copy()
        tmp_first_cooc = self.dist_mat.copy()
        tmp_cooc_tensor = self.cooc_mat.copy()
        tmp_timeline = self.timeline.copy()
        tmp_G = self.graph.copy()
        tmp_first_occs = self.first_occs.copy()
        tmp_text = self.text.copy()
        # shuffle the text
        np.random.shuffle(self.text)
        self.windows = self.text
        t = self.cooc_2d()
        new_cutoff = self.cutoff
        self.index = tmp_index
        self.index_labels = tmp_index_labels
        self.first_occs = tmp_first_occs
        self.graph = tmp_G
        self.dist_mat = tmp_first_cooc
        self.cooc_mat = tmp_cooc_tensor
        self.timeline = tmp_timeline
        self.text = tmp_text
        return t, new_cutoff

    def oaat_filtration(self, dist_mat):
        """
        Takes as input a filtratino matrix ``dist_mat`` which does not
        necessarily introduce edges one at a time (i.e., with sentence-level
        granularity, so their may be some redundancy of values in the matrix)
        and returns a filtration which does so. Useful for "unfurling"
        a filtration. See paper for more details.
        """
        oaat_mat = np.full_like(dist_mat, np.inf)
        maxval = np.max(dist_mat[np.isfinite(dist_mat)])  # max value
        count = 1
        rel_timeline = {}
        # iterate sentence by sentence
        for v in range(1, int(maxval) + 1):
            indices = list(zip(*np.where(dist_mat == v)))
            # ensure we change (i, j) and (j, i) spots simultaneously
            indices = list(set(tuple(sorted(i)) for i in indices))

            nodes = [x for x in indices if x[0] == x[1]]
            edges = [x for x in indices if x[0] != x[1]]
            # randomly shuffle order of introduction for oaat
            np.random.shuffle(nodes)
            np.random.shuffle(edges)
            # introduce all nodes first, in random order
            for node in nodes:
                oaat_mat[node] = count
                rel_timeline[count] = self.index_labels[node[0]]
                count += 1
            # then introduce edges
            for edge in edges:
                for ind in [edge, tuple(reversed(edge))]:
                    oaat_mat[ind] = count
                    rel_timeline[count] = (self.index_labels[ind[0]],
                                           self.index_labels[ind[1]])
                count += 1
        return oaat_mat, count, rel_timeline

    def node_ordered_filtration(self):
        """
        Creates a node-ordered filtration, where nodes and all their edges to
        previously-introduced nodes are added in order of node introduction,
        and for any given node, edges are added in random order. See paper for
        more details.
        """
        nd_dist_mat = np.full_like(self.dist_mat, np.inf)
        fos = sorted(set(self.first_occs)) # unique values so we can randomize addition
        count = 1
        introduced_inds = []
        # loop through values in first occurrences
        for v in fos:
            inds = np.where(self.first_occs == v)[0]
            # randomly shuffle if there are multiple - if introduced in the same sentence,
            # we want a random order
            np.random.shuffle(inds)
            for i in inds:
                introduced_inds.append(i)
                nd_dist_mat[i, i] = count
                count += 1
                # go through all the previously-introduced indices/concepts
                allowed_prev = np.array(introduced_inds[:-1])
                np.random.shuffle(allowed_prev)
                # and introduce connections with those previous ones in a random order
                for j in allowed_prev:
                    if self.dist_mat[i, j] != np.inf:
                        nd_dist_mat[i, j] = count
                        nd_dist_mat[j, i] = count
                        count += 1
        return nd_dist_mat, count

    def rnd_edge_filtration(self):
        """
        Creates a random edge filtration, in which edges from the final network
        are added in a random order. See paper for more details.
        """
        G = self.graph
        nodes = list(G.nodes)
        edges = list(G.edges)
        np.random.shuffle(edges)
        A = np.full((len(nodes), len(nodes)), np.inf)
        rel_timeline = {}
        count = 1
        for edge in edges:
            i, j = nodes.index(edge[0]), nodes.index(edge[1])
            # make sure corresponding nodes are introduced prior to introducing edge
            for ind in [i, j]:
                if A[ind, ind] == np.inf:
                    A[ind, ind] = count
                    count += 1
            A[i, j] = count
            A[j, i] = count
            rel_timeline[count] = edge
            count += 1
        return A, count, rel_timeline

    def get_barcode(self, filt_mat, maxdim=2):
        """
        Calculates the persistent homology for a given filtration matrix
        ``filt_mat``, default dimensions 0 through 2. Wraps ripser.
        """
        b = ripser(filt_mat, distance_matrix=True, maxdim=maxdim)['dgms']
        return list(zip(range(maxdim+1), b))

    """
    Old unused code regarding pointwise mutual information as a potential
    alternative for judging relatedness of keyphrases/index phrases/concepts

    # def PPMI(self):
    #     if self.cooc_mat is None:
    #         raise ValueError('No cooccurrence matrix')
    #     pmi = np.zeros((len(self.cooc_mat), len(self.cooc_mat)))
    #     for i in range(len(pmi)):
    #         for j in range(i, len(pmi)):
    #             num_i, num_j = self.cooc_mat[i, i], self.cooc_mat[j, j]
    #             num_cooc = self.cooc_mat[i, j]
    #             N = len(self.windows) # number of sentences
    #             npmi_val = np.log(num_cooc*N/(num_i*num_j))/(-np.log(num_cooc/N))
    #             pmi[i, j] = npmi_val
    #             pmi[j, i] = npmi_val
    #     pmi[np.isnan(pmi)] = 0
    #     pmi[pmi <= 0] = 0
    #     return pmi

    # def PPMI_graph(self):
    #     m = self.PPMI()
    #     graph = nx.from_numpy_array(m)
    #     mapping = {i: label for i, label in enumerate(self.index_labels)}
    #     nx.relabel_nodes(graph, mapping, copy=False)
    #     graph.remove_edges_from(nx.selfloop_edges(graph))
    #     return graph

    # def PPMI_filtration(self):
    #     m = self.PPMI()
    #     pmi_dis_mat = self.dist_mat.copy()
    #     pmi_dist_mat[m <= 0] = np.inf
    #     return pmi_dist_mat
    """

    """
    Old unused code for a topological distance filtration null model

    # topological distance filtration; adds nodes in by distance from first-introduced
    # concept, where edge distance = 1/weight, since weight denotes strength.
    # at each step, adds each node's edges to already-added nodes *only* in order of decreasing
    # weight - from strongest to weakest connections
    # some element of stochasticity due to ordering for equidistant/equal-weight nodes/edges
    def topo_dist_filtration(self):
        filt_mat = np.full((len(self.index_labels), len(self.index_labels)), np.inf) # eventual filtration
        # edge pool
        edges = set(self.graph.edges)
        all_nodes = list(self.graph.nodes)
        # nodes that have been added already
        added_nodes = {self.index_labels[0]}
        # get shortest path distances:
        dists = nx.single_source_dijkstra_path_length(self.graph,
                                                      source=self.index_labels[0],
                                                      weight=lambda u, v, d: 1/d['weight']
                                                      if d['weight'] != 0 else None)
        # all possible distance values - necessary for randomization in case of equality
        vals = sorted(set(dists.values()))[1:]
        n = 1
        for val in vals:
            # get nodes with the given distance value, randomly shuffle them
            nodes = [n for n in dists.keys() if dists[n] == val]
            np.random.shuffle(nodes)
            for node in nodes:
                # add node, get the edges associated with the added nodes
                added_nodes.add(node)
                # add node birth to filtration matrix
                filt_mat[all_nodes.index(node), all_nodes.index(node)] = n
                n += 1
                # now look at the allowed edges
                allowed_edges = [e for e in edges if e[0] in added_nodes and e[1] in added_nodes]
                edges.difference_update(allowed_edges)
                # get unique weight values - again important for randomization
                weight_vals = sorted(set([self.graph.edges[ae]['weight'] for ae in allowed_edges]), reverse=True)
                for wv in weight_vals:
                    wv_edges = [e for e in allowed_edges if self.graph.edges[e]['weight'] == wv]
                    np.random.shuffle(wv_edges)
                    for edge in wv_edges:
                        i, j = all_nodes.index(edge[0]), all_nodes.index(edge[1])
                        filt_mat[i, j] = n
                        filt_mat[j, i] = n
                        n += 1
        return filt_mat, n
    """

    """
    Old unused code for a connected filtration (where edges contributed to a
    single connected component). May be incorrect.

    # creates a filtration matrix where, starting from a random node, edges are added
    # that only contribute to the single connected component (until the end, when the
    # stragglers are added in)
    # old and not used
    # def connected_filtration(self):
    #     G = self.graph
    #     nodes = list(G.nodes)
    #     edges = list(G.edges)
    #     first_node = np.random.choice(nodes)
    #     allowed_nodes = {first_node}
    #     A = np.full((len(nodes), len(nodes)), np.inf)
    #     n = 1 # value we set the edge to; increment after each one
    #     while edges != []:
    #         allowed_edges = list(filter(lambda e: not set(e).isdisjoint(allowed_nodes), edges))
    #         if allowed_edges == []:
    #             allowed_edges = edges
    #         edge = random.choice(allowed_edges)
    #         edges.remove(edge)
    #         allowed_nodes.update(set(edge))
    #         i, j = nodes.index(edge[0]), nodes.index(edge[1])
    #         A[i, j] = n
    #         A[j, i] = n
    #         n += 1
    #     for i in range(len(nodes)):
    #         A[i, i] = 0
    #     print("Number of edges: " + str(n) + " (for filtration cutoff)")
    #     return A, n
    """

    """
    Old unused code for a degree filtration, where nodes are added in order of
    degree and their edges are added in order of decreasing degree
    (so "strongest"/"most important" connections and topics are added earlier)

    # creates a filtration matrix based on degree of nodes
    # adds nodes in order of decreasing degree, and edges in order of decreasing weight
    def degree_filtration(self):
        G = self.graph
        nodes = list(G.nodes)
        edges = set(G.edges(data='weight'))
        degs = sorted(list(dict(nx.degree(G)).items()), key=lambda x: x[1], reverse=True)
        A = np.full((len(nodes),len(nodes)), np.inf)
        n = 1
        #print(degs)
        while len(edges) > 0:
            node = degs.pop(0)[0]
            allowed_edges = list(filter(lambda e: node in e, edges))
            edges = edges.difference(set(allowed_edges))
            # random ordering
            allowed_edges = sorted(allowed_edges, key=lambda x: x[2], reverse=True)
            #print(allowed_edges)
            for edge in allowed_edges:
                i, j = nodes.index(edge[0]), nodes.index(edge[1])
                A[i, j] = n
                A[j, i] = n
                n += 1
        for i in range(len(nodes)):
            A[i, i] = 0
        return A, n
    """

    """
    Old unused code for plotting barcodes, and annotating with the words/edges
    representing the born and dead cycles.

    def plot_barcode(self, bars, dims=[1,2], length=None, k=None, labels=None):
        plt.figure(figsize=(12/2.54, 4), dpi=300)
        colors = ['b', 'c', 'g']
        count = 1
        if length is not None:
            cutoff = length
        elif labels is not None:
            cutoff = max(labels.keys()) + 1
        else:
            cutoff = self.cutoff
        bars = dict(bars)
        has_inf = False # notes whether there are infinitely-persisting cycles
        for d in dims:
            try:
                bn = bars[d]
            except KeyError:
                print('Dimension not in barcode!')
            color = colors.pop(0) #better not have any more than 3 dimensions
            bn = sorted(bn, key=lambda x:x[0])
            for b, i in zip(bn, range(len(bn))):
                if b[1] == np.inf:
                    has_inf = True
                    b = (b[0], 1.5*cutoff) # arbitrary, so it overhangs
                if i == 0:
                    plt.plot(b, [count, count], color=color,
                             label='Dimension {}'.format(d))
                else:
                    plt.plot(b, [count, count], color=color,
                             label=None)
                if labels == 'edge':
                    if b[1] - b[0] > cutoff/10 or b[1] == 1.5*cutoff:
                        f = lambda e: e[0] + '->' + e[1]
                        edge_1 = self.rel_timeline[b[0]]
                        plt.annotate(f(edge_1), (b[0]-1, count-0.4), horizontalalignment='right',fontsize=8)
                        if b[1] != 1.5*cutoff:
                            edge_2 = self.rel_timeline[b[1]]
                            plt.annotate(f(edge_2), (b[1]+1, count-0.4), horizontalalignment='left',fontsize=8)
                elif labels != None: # requires a dict of edges
                    if b[1] - b[0] > cutoff/10 or b[1] == 1.5*cutoff:
                        f = lambda e: e[0] + '->' + e[1]
                        edge_1 = labels[b[0]]
                        plt.annotate(f(edge_1), (b[0]-1, count-0.4), horizontalalignment='right',fontsize=8)
                        if b[1] != 1.5*cutoff:
                            edge_2 = labels[b[1]]
                            plt.annotate(f(edge_2), (b[1]+1, count-0.4), horizontalalignment='left',fontsize=8)
                count += 1
        if has_inf:
            plt.xlim(0, 1.1*cutoff)
        #plt.axvline(x=cutoff, color='r', linestyle='--', label='End of Filtration')
        if k is not None:
            plt.axvline(x=k, color='m', linestyle='--', label='End of Connected Component')
        plt.legend(loc='lower right', fontsize=8)
        plt.show()
    """

    """
    Old, unused code wrapping barcode generation and plot for the expositional
    network

    # Plots the barcode for the expositional ordering network
    def plot_expos_barcode(self, dims=[1,2], labels='edge'):
        # if self.ph_dict exists, it'll use that; otherwise, it'll make it
        bars, _ = self.get_barcode(dims=dims)
        self.plot_barcode(bars, dims=dims, labels=labels)
    """
