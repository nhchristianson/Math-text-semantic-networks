import nltk
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
import networkx as nx
import enchant
import numpy as np
from scipy import stats as st
import string
import matplotlib.pyplot as plt
import random
import community
from itertools import *
from tqdm import tqdm_notebook
import re
# from ConceptExtraction import *
import RAKE
from ripser import ripser

# cleaning and corpus stuff
import spacy
from nltk.corpus import brown
nlp = spacy.load('en', disable=['parser', 'ner'])


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

# Import our Julia stuff
# import julia
# j = julia.Julia()
# from julia import Eirene

# useful things
d = enchant.Dict("en_US") #our spell-checker
ps = SnowballStemmer('english') #our stemming algo
wnl = WordNetLemmatizer()
# stopwords1 = set(stopwords.words('english'))
stop_words = [x for x in RAKE.RanksNLLongStopList() if len(x) > 1 or x == 'a'] # previous RAKE NL Long
stop_words.remove('value')
vowels = {'a', 'e', 'i', 'o', 'u', 'y'}


def freq_wholelabel(ind_lst, txt):
    freqs = []
    for ind in ind_lst:
        freqs.append(txt.count(ind))
    return {k: f for k, f in zip(ind_lst, freqs)}


"""
Class allowing for the construction of semantic networks from text
note that all(?) indices here are 1-indexed (to differentiate between the
beginning of a text, index 0, and the first sentence, index 1)

NOTE: REQUIRES PYTHON PACKAGE PYJULIA, AND REQUIRES A JULIA INSTALLATION,
WITH PYCALL AND EIRENE INSTALLED.
ENSURE THAT PYCALL IS DIRECTED AT THE INSTALLATION OF PYTHON YOU ARE CURRENTLY
USING, AND BE SURE NOT TO INSTALL JULIAPRO - IT DOESN'T WORK WITH PYJULIA
FOR SOME REASON

"""
class Concept_Network:

    def __init__(self):
        #self.SIZE = SIZE deprecated, don't use this anymore.
        self.index = None
        self.text = None # text should not change
        self.windows = None # windows is sliding windows based on text
        self.dist_mat = None
        self.cooc_mat = None
        self.graph = None
        self.index_labels = None #Recall that eventually, index labels will be sorted by introduction
        self.timeline = None # gives sentence number of edge origin
        self.cutoff = None
        self.bars = []
        self.ph_dict = None # Our Eirene persistent homology dict class
        self.conn_comps = None
        self.modules = None
        self.cooc_mat = None
        self.weight_dist = None

    ### All old text cleaning utility stuff
    ### Deprecated since RAKE implementation

    # # Returns a cleaned version of the index
    # # This is a 2d list where each element is a list of terms of an index element
    # def clean_index(self, filename, stem=True):
    #     with open(filename) as f:
    #         index = f.readlines()
    #     index = [re.sub("\u2013|\u2014", "-", i) for i in index]
    #     index = [i.split() for i in index]
    #     index = [[w.translate(translator) for w in i] for i in index]
    #     index = [[w for w in i if re.search('[^a-zA-Z-]+', w) is None] for i in index]
    #     index = [[w.lower() for w in i] for i in index]
    #     index = [[w for w in i if w != 'index' and w != 'subject'] for i in index]
    #     index = [[w for w in i if len(w) > 1] for i in index]
    #     index = [[w for w in i if len(w) > 2 or w in self.rm_exceptions] for i in index]
    #     index = [[w for w in i if d.check(w) or len(w) > 3 or w in self.rm_exceptions] for i in index]
    #     index = [i for i in index if i != []]
    #     #ensure no repeats
    #     if stem:
    #         index = [[ps.stem(w) if w not in self.stem_exceptions else wnl.lemmatize(w) for w in i] for i in index]
    #     else:
    #         index = [[wnl.lemmatize(w) for w in i] for i in index]
    #     index = list(map(lambda l: ' '.join(l), index))
    #     index = list(set(index))
    #     index = list(map(lambda l: l.split(' '), index))
    #     self.index = index
    #     return index

    # # naive approach to index building
    # def clean_index_naive(self, filename, stem=True):
    #     with open(filename) as f:
    #         index = f.read()
    #     index = re.sub("\u2013|\u2014", "-", index) # replace problematic unicode
    #     index = word_tokenize(index)
    #     index = [i.translate(translator) for i in index]
    #     index = [i for i in index if re.search('[^a-zA-Z-]+', i) is None]
    #     index = [i.lower() for i in index]
    #     index = [i for i in index if i not in self.rm]
    #     index = [i for i in index if i != 'index'] # not i in stop_words and
    #     index = [i for i in index if len(i) > 1]
    #     index = [i for i in index if len(i) > 2 or i in self.rm_exceptions]
    #     index = [i for i in index if d.check(i) or len(i) > 3 or i in self.rm_exceptions]
    #     if stem:
    #         index = [ps.stem(i) if i not in self.stem_exceptions else wnl.lemmatize(i) for i in index]
    #     else:
    #         index = [wnl.lemmatize(i) for i in index]
    #     index = list(set(index))
    #     index = list(map(lambda x: [x], index))
    #     self.index = index
    #     return index

    # # Cleans a chapter
    # # index input not actually used atm
    # def clean_chapter(self, filename, index=None, stem=True):
    #     with open(filename) as chapter:
    #         text =  chapter.read()
    #     text = re.sub("\u2013|\u2014", "-", text)
    #     sents = sent_tokenize(text) # Split into sentences
    #     sents = [word_tokenize(s) for s in sents]
    #     sents = [[w.translate(translator) for w in s] for s in sents] # filter punctuation
    #     sents = [[w for w in s if re.search('[^a-zA-Z-]+', w) is None] for s in sents] #strips everything but alphabetic
    #     sents = [[w.lower() for w in s] for s in sents] # make lower case
    #     #sents = [[w for w in s if w not in self.rm] for s in sents]
    #     #sents = [[w for w in s if not w in stop_words] for s in sents] # filter stop words
    #     # basically filters out all our weird variables and stuff like that
    #     # # without filtering out legit stuff
    #     #sents = [[w for w in s if len(w) > 1] for s in sents] #filters out variables, etc
    #     #sents = [[w for w in s if len(w) > 2 or w in self.rm_exceptions] for s in sents]
    #     #sents = [[w for w in s if d.check(w) or len(w) > 3 or w in self.rm_exceptions] for s in sents]
    #     sents = [s for s in sents if len(s) > 0]
    #     """
    #     Removed so we don't lose things that don't check but are legit
    #     if index: # if we have an index list, check if spellcheck or if in index
    #         index = set(index)
    #         words = [w for w in words if d.check(w) or w in flat_index]
    #     else:
    #         words = [w for w in words if d.check(w)] # check if word spellchecks
    #     """
    #     if stem:
    #         words = [[ps.stem(w) if w not in self.stem_exceptions else wnl.lemmatize(w) for w in s] for s in sents]
    #     else:
    #         words = [[wnl.lemmatize(w) for w in s] for s in sents]
    #     return words

    # # cleans chapters and puts them together as one list of strings
    # # separates them by a space of 100 empty strings to preserve independence
    # # if printindices is True, will print the indices where chapters end
    # # expects chapters formatted as 'chapter_n.txt'
    # def clean_chapters(self, folder, numchapters, stem=True, index=None, printindices=False):
    #     all_text = []
    #     for n in range(1, numchapters+1):
    #         all_text += self.clean_chapter(folder + '/chapter_{}.txt'.format(n), stem=stem)
    #     self.text = all_text

    def clean(self, textname, n_chapt):
        alltext = []
        for i in range(1, n_chapt+1):
            with open('Textbooks/{0}/chapter_{1}.txt'.format(textname,
                                                             i)) as chapter:
                text =  chapter.read()
            import unicodedata
            text = unicodedata.normalize('NFKD',
                                         text).encode('ascii','ignore')
            text = b' '.join(text.split()).decode()
            text = text.replace('-', ' ')
            text = nlp(text)
            text = ' '.join(w.lemma_ for w in text)
            sents = sent_tokenize(text)
            sents = [word_tokenize(s) for s in sents]
            # replace all things with numbers with #
            sents = [[w if not re.search(r"\d+", w) else '#' for w in s]
                     for s in sents]
            # remove most things
            sents = [[w for w in s if re.search('[^a-zA-Z-#]+', w) is None]
                     for s in sents]
            sents = [[w.lower() for w in s] for s in sents]
            text = sents
            # replace all potential variables with VAR
            # text = [nlp(' '.join(s)) for s in text]
            # text = [[w.lemma_ for w in s] for s in text]
            # text = [[ps.stem(w) for w in s] for s in text]
            # make sure we have some vowely-bois
            def quick_check(word):
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
            text = [[w if quick_check(w) else 'VAR' for w in s] for s in text]
            alltext += text
        #alltext = [list(filter(lambda x: '.' not in x, s)) for s in alltext]
        self.text = list(filter(lambda x: len(x) > 0, alltext))
        self.windows = self.text
        return self.text

    def RAKE_keywords(self):
        extra_sl = ['example', 'counterexample', 'text', 'texts', 'undergraduate', 'chapter', 'definition', 'notation',
                    'proof', 'exercise', 'result']
        sl = nlp(' '.join(stop_words + extra_sl))
        sl = [w.lemma_ for w in sl]
        rake = RAKE.Rake(sl + ['VAR', '#', '-pron-', '-pron'])
        text = '. '.join([' '.join(x) for x in self.text])
        kws = rake.run(text, minCharacters=3, maxWords=4, minFrequency=5)
        keyphrases = [' '.join([j for j in k[0].split(' ')
                      if j not in {'var', '#', '-pron-', '-pron'}]) for k in kws]
        #print(*enumerate([k for k in kws if 'var' not in k[0] and '#' not in k[0]]), sep='\n')
        keyphrases = set([k for k in keyphrases if len(k) > 0])
        index = list(map(lambda x: x.split(), keyphrases))
        self.index = index
        return index


    # determines order of first occurrence of index terms.
    # prunes out the index elements which do not occur within the text
    # (i.e., within a single sentence), and orders them w.r.t. first occurrence
    # very very important to run before doing anything else
    def occurrence(self, debug=False, n_allowed=None):
        sentences = self.text
        from collections import Counter
        index = list(map(set, self.index))
        index_labels = np.array(list(map(lambda i: ' '.join(i), self.index)))
        #print(index_labels)
        first_occs = [] # sentence number of first term occurrence
        counts = []
        conn_text = '. '.join([' '.join(s) for s in sentences])
        # print(len(sentences))
        for ind in index_labels:
            # print(ind)
            for sent, n in zip(sentences, range(1,len(sentences)+1)):
                #print(sent)
                if ind in ' '.join(sent):
                    first_occs.append(n)
                    break
            else:
                first_occs.append(0)
            # count how many times index term appears in the text
            counts.append(conn_text.count(ind))
        # print(counts)
        counts = sorted(list(zip(index_labels, counts)), key=lambda x: x[1], reverse=True)
        ordered = [c[0] for c in counts]
        first_occs = np.array(first_occs)
        # print(first_occs)
        nonzeros = np.nonzero(first_occs)[0]
        zeros = np.where(first_occs == 0)[0]
        print("Yield percent:", len(nonzeros)/len(first_occs)) # quick metric for success rate
        if debug:
            print(index_labels[zeros])
        #remove terms which do not occur
        index = np.array(index)[nonzeros]
        index_labels = index_labels[nonzeros]
        #sort remaining into order of introduction
        sort = np.argsort(first_occs[nonzeros])
        index = index[sort]
        index_labels = index_labels[sort]
        index_labels = np.array([x for x in index_labels if x in ordered[:n_allowed]])
        index = np.array(list(map(lambda i: set(i.split()), index_labels)))
        self.first_occs = np.sort(first_occs[nonzeros])
        self.index_labels = index_labels
        self.index = index


    # generates the index by extracting concepts using the ConceptExtraction
    # module
    # def extract_concepts(self, stem=False, stemresult=False, niter=100):
    #     thresholds = [[5, 5, 5, 5, 5]] + [[0.3, 0.4, 0.5, 0.6, 0.7]] * 4
    #     dom_thres = [1.5]*5
    #     sw = {'exercise','show','let','use','find','sect','see','place',
    #           'defin','hint', 'work', 'case', 'also', 'claim',
    #           'section', 'example', 'respect', 'next', 'note', 'ha', 'nothing',
    #           'make', 'like', 'following', 'way', 'follow', 'whose', 'chapter',
    #           'question', 'much', 'one', 'two', 'three', 'four', 'iii', 'first',
    #           'second', 'third', 'last', 'important', 'select', 'quite',
    #           'springer', 'publish'}
    #     sig_words = set().union(*self.index)
    #     CE = ConceptExtraction(sum(self.text, []), thresholds, dom_thres,
    #                            sig_words=sig_words, stopwords=sw, stem=stem)
    #     guess = [0.3, 0.4, 0.5, 0.6, 0.7]*4 + [1.5]*5
    #     x, seen = CE.params_optimize(guess=guess, niter=niter)
    #     print(x) #, seen)
    #     CE.conceptExtraction()
    #     # CE.wordConceptExtraction()
    #     concepts = list(set(CE.ngram_concepts))# + CE.word_concepts))
    #     concepts = list(set(concepts).intersection(set(map(lambda s: tuple(s.split()), seen))))
    #     #concepts = [tuple(s.split()) for s in seen]
    #     concept_sets = [set(c) for c in concepts]
    #     repeated_indices = []
    #     for i in range(len(concept_sets)):
    #         if concept_sets[i] in concept_sets[:i]:
    #             repeated_indices.append(i)
    #     concepts = np.array(concepts)
    #     np.delete(concepts, repeated_indices)
    #     # ordering is important so as to ensure that index labels are ordered correctly
    #     index_labels = np.array(list(map(lambda i: ' '.join(i), concepts)))
    #     self.index_labels = index_labels
    #     if stemresult:
    #         concepts = [{ps.stem(w) for w in c} for c in concepts]
    #     self.index = np.array(concepts)

    # generates a list of sliding windows of size "size" from a sentence
    def sliding_window(self, lst, size):
        windows = []
        for i in range(0, max(1, len(lst)-size+1)):
            b, e = i, min(i+size, len(lst))
            windows.append(lst[b:e])
        return windows

    # gets all sliding windows of size "size" from a list of sentences
    def sliding_window_constr(self, size):
        windows = []
        if self.text is not None:
            for sent in self.text:
                windows += self.sliding_window(sent, size)
        self.windows = windows


    # more efficient co-occurence function for 2nd order cooc
    # should be more efficient than the generalized case.
    def cooc_2d(self):
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
        # make graph for some reason
        G = nx.from_numpy_array(cooc_mat)
        G.remove_edges_from(nx.selfloop_edges(G))
        name_mapping = {i: label for i, label in enumerate(self.index_labels)}
        nx.relabel_nodes(G, name_mapping, copy=False)
        self.graph = G
        return cooc_mat, first_cooc, timeline

    # returns cooccurrence count and time matrices/tensors for nth order cooccurrences
    # where n gives the cooccurrence order
    # (i.e., we're looking for cooccurrences of n elements)
    # effectively works recursively; i.e., uses the previous order cooccurrence to
    # build the next order efficiently, by ignoring things which don't cooccur at that
    # previous order
    # This is the one
    # conn comps and modules fairly self-explanatory, only do those if looking at second order cooccurrence
    # threshold keeps all connections >= the value of 'threshold'
    # THIS ACTUALLY DOUBLE COUNTS OCCURRENCES!!!!!
    # def n_order_cooc(self, n, prev_cooc=None, conn_comps=False, modules=False, groups=None, threshold=2):
    #     if n < 2:
    #         raise ValueError("Can't calculate cooccurrence of single terms!")
    #     if n != 2:
    #         conn_comps, modules = False, False
    #     dims = (len(self.index),) * n
    #     if prev_cooc is None:
    #         prev_cooc = np.ones(dims[:-1])# if not supplied with a previous cooccurrence matrix, just assume everything is there (not efficient!)
    #     cooc_tensor = np.zeros(dims)
    #     first_cooc = np.zeros(dims)
    #     prev_cooc = np.transpose(np.nonzero(prev_cooc)) # these three lines give us the "places" to look
    #     prev_cooc = list(map(sorted, prev_cooc))        # in the 2nd order case, doesn't really do anything
    #     prev_cooc = set(map(tuple, prev_cooc))          # should be super helpful for higher orders
    #     sentences = self.windows #list(map(set, self.windows))
    #     timeline = {}
    #     make_graph = conn_comps or modules
    #     G = nx.Graph()
    #     comps, mods, avg_shortest_path = [], [], []
    #     record_communities = groups is not None and len(set(groups)) > 2
    #     if record_communities:
    #         group_edge_hist = [[0] for _ in range(max(groups)+1)]
    #     for sent, num in tqdm_notebook(list(zip(sentences, range(1, len(sentences)+1)))):
    #         if record_communities:
    #             i_ = [0 for _ in range(max(groups)+1)]
    #         for comb in prev_cooc:
    #             #if all(item.issubset(sent) for item in self.index[list(comb)]):
    #             if all(item in ' '.join(sent) for item in self.index_labels[list(comb)]):
    #                 G.add_node(self.index_labels[comb[0]])
    #                 for i in range(max(comb), len(self.index)):
    #                     # if self.index[i].issubset(sent):
    #                     if self.index_labels[i] in ' '.join(sent):
    #                         #G.add_node(self.index_labels[i])
    #                         #G.add_edge(self.index_labels[comb[0]], self.index_labels[i])
    #                         multi_ind = comb + (i,)
    #                         if record_communities and multi_ind[0] != multi_ind[1]:
    #                             if groups[multi_ind[0]] == groups[multi_ind[1]]:
    #                                 i_[groups[multi_ind[0]]] += 1
    #                             else:
    #                                 i_[0] += 1
    #                         labels = self.index_labels[list(multi_ind)]
    #                         indices = list(permutations(multi_ind))
    #                         for idx in indices:
    #                             cooc_tensor[idx] += 1
    #                         if first_cooc[multi_ind] == 0:
    #                             for idx in indices:
    #                                 first_cooc[idx] = num
    #                             if len(set(multi_ind)) == n: # this way we only record if it's a real n-clique
    #                                 timeline[tuple(labels)] = num
    #         if conn_comps:
    #             comps.append(nx.number_connected_components(G))
    #             try:
    #                 conn_comp_subs = nx.connected_component_subgraphs(G)
    #                 mean = np.max([nx.average_shortest_path_length(Gi) for Gi in conn_comp_subs])
    #                 avg_shortest_path.append(mean)
    #             except:
    #                 avg_shortest_path.append(0)
    #         if modules:
    #             mods.append(len(set(community.best_partition(G).values())))
    #         if record_communities:
    #             for j in range(len(i_)):
    #                 group_edge_hist[j].append(group_edge_hist[j][-1]+i_[j])
    #     if conn_comps:
    #         self.conn_comps = comps
    #         self.avg_shortest_path = avg_shortest_path
    #     if modules:
    #         self.modules = mods
    #     if record_communities:
    #         self.community_dev = group_edge_hist
    #     #G.remove_edges_from(nx.selfloop_edges(G))
    #     #self.graph = G
    #     it = np.nditer(first_cooc, flags=['multi_index'], op_flags=['readwrite'])
    #     # make sure off-"diagonal" zeros are infinity (connections that are never born)
    #     # and "diagonal" entries are zero.
    #     if n == 2:
    #         first_occs = []
    #         for i in range(len(first_cooc)):
    #             first_occs.append(first_cooc[i, i])
    #         self.first_occs = first_occs
    #     while not it.finished:
    #         if len(set(it.multi_index)) == n and it[0] == 0:
    #             first_cooc[it.multi_index] = np.inf
    #         elif len(set(it.multi_index)) < n:
    #             first_cooc[it.multi_index] = 0
    #         it.iternext()
    #     self.cutoff = len(sentences)
    #     # thresholding step
    #     first_cooc[cooc_tensor < threshold] = np.inf
    #     np.fill_diagonal(first_cooc, 0)
    #     cooc_tensor[cooc_tensor < threshold] = 0
    #     if n == 2:
    #         self.dist_mat = first_cooc
    #         self.cooc_mat = cooc_tensor
    #         self.timeline = timeline
    #         G = nx.from_numpy_array(cooc_tensor)
    #         G.remove_edges_from(nx.selfloop_edges(G))
    #         name_mapping = {i: label for i, label in enumerate(self.index_labels)}
    #         nx.relabel_nodes(G, name_mapping, copy=False)
    #         self.graph = G
    #     return cooc_tensor, first_cooc, timeline

    # # Basically does everything for calculating pairwise coocurrences, with the
    # # added benefit of saving concepts so they only ever have to be computed once.
    # def cooc_for_text(self, text_name, nchapts, niter=100, stem=False, stemresult=False, conn_comps=False, modules=False, groups=None):
    #     self.clean_chapters('Textbooks/'+text_name, nchapts, stem=stem)
    #     self.clean_index_naive('Textbooks/'+text_name+'/index.txt', stem=stem)
    #     self.occurrence()
    #     # You'd better pray that if there are pre-existing concepts, they were
    #     # created with the same formatting you're using now...
    #     try:
    #         concepts = np.load('Textbooks/'+text_name+'/concepts.npy')
    #         self.index_labels = concepts
    #         self.index = np.array(list(map(lambda c: set(c.split()), concepts)))
    #     except FileNotFoundError:
    #         self.extract_concepts(stem=stem, stemresult=stemresult, niter=niter)
    #         self.occurrence()
    #         np.save('Textbooks/'+text_name+'/concepts.npy', self.index_labels)
    #     #self.sliding_window_constr(10)
    #     self.windows = self.text
    #     t = self.n_order_cooc(n=2, conn_comps=conn_comps, modules=modules, groups=groups)
    #     return t

    # new cooc_for_text with new cleaning, etc.
    def cooc_for_text(self, text_name, nchapts, threshold=1, n_allowed=None, conn_comps=False, modules=False, groups=None):
        self.clean(text_name, nchapts)
        self.RAKE_keywords()
        self.occurrence()
        brown_whole = freq_wholelabel(self.index_labels, bwords)
        # do our new augmented rake-IDF scoring
        # first re-run rake, just for convenience sake
        # since we just ran "occurrence" and this is useful
        extra_sl = ['example', 'counterexample', 'text', 'texts', 'undergraduate', 'chapter', 'definition', 'notation',
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
            #score /= math_whole[ind][0]/math_whole[ind][1]
            # trying with baseline being 1
            brownscore = brown_whole[ind] + 1
            # if brownscore == 0:
            #     brownscore = 0.01
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

    def cont_config_graph(self):
        nodes = list(self.graph.nodes)
        degs = dict(self.graph.degree())
        dT = sum(degs.values())
        strengths = {n: sum([x[2]['weight'] for x in self.graph.edges(n, data=True)])
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
            DISTRIBUTIONS = [st.pareto, st.lognorm, st.levy, st.dweibull, st.burr,
                             st.fisk, st.loggamma, st.loglaplace, st.powerlaw]
            results = []
            normedweights = np.array(normedweights)
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
        for i in range(1, len(nodes)):
            for j in range(i+1, len(nodes)):
                d_uv = degs[nodes[i]]*degs[nodes[j]]/dT
                if np.random.rand() < d_uv:
                    s_uv = strengths[nodes[i]]*strengths[nodes[j]]/sT
                    xi = self.weight_dist.rvs()
                    null_graph[i, j] = xi*s_uv/d_uv
                    null_graph[j, i] = xi*s_uv/d_uv
        return null_graph

    # returns a filtration matrix for the cooccurrence filtration of the text
    # with a set of randomly-selected words from the body of the text
    def random_index_null(self, return_index=False, return_otherstuff=False):
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

    # returns a filtration matrix and stuff for the cooccurrence filtration of the text
    # with the same index, but sentences randomly shuffled
    def rnd_sent_ord_null(self):
        tmp_index = self.index.copy()
        tmp_index_labels = self.index_labels.copy()
        tmp_first_cooc = self.dist_mat.copy()
        tmp_cooc_tensor = self.cooc_mat.copy()
        tmp_timeline = self.timeline.copy()
        tmp_G = self.graph.copy()
        tmp_first_occs = self.first_occs.copy()
        tmp_text = self.text.copy()
        # gotta shuffle the text
        np.random.shuffle(self.text)
        # print(self.text)
        # self.occurrence()
        self.windows = self.text
        t = self.cooc_2d()
        new_first_occs = self.first_occs.copy()
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

    # takes in a filtration matrix which does not introduce edges in one at a
    # time (i.e., with sentence-level granularity) and returns a filtration
    # matrix which does so. Is useful for seeing which connections begin/kill
    # persistent cycles in the barcode. Also returns a "time"-edge dict
    # adds edges in a particular sentence randomly - need to average
    def oaat_filtration(self, dist_mat):
        oaat_mat = np.full_like(dist_mat, np.inf)
        maxval = np.max(dist_mat[np.isfinite(dist_mat)]) # max value
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
            #print(len(nodes), len(edges))
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


    # creates a node-ordered one
    # order given by order of introduction, with ties broken by random shuffle
    # First a node is introduced, then its connections to previously-introduced nodes
    # are introduced in a random order
    def node_ordered_filtration(self):
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

    # creates a randomly-ordered edge filtration matrix for the network
    # nodes are introduced immediately before they are first included in an edge
    def rnd_edge_filtration(self):
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

    # creates a filtration matrix based on the distance of nodes from the initial topic
    # note that this acts by calculating all distances from the source node, then, in order
    # of other nodes' distances, adding them (and *all* of their edges) in
    # def dist_filtration(self):
    #     G = self.graph
    #     nodes = list(G.nodes)
    #     edges = list(G.edges)
    #     first_node = self.index_labels[0]
    #     A = np.full((len(nodes),len(nodes)), np.inf)
    #     n = 1
    #     k = None # this'll be where we store when we start adding unconnected components
    #     paths = list(nx.shortest_path_length(G, first_node).items())
    #     paths = sorted(paths, key=lambda i: i[1])
    #     while edges != []:
    #         if len(paths) == 0:
    #             k = n + 1
    #             for edge in edges:
    #                 i, j = nodes.index(edge[0]), nodes.index(edge[1])
    #                 A[i, j] = n + 1
    #                 A[j, i] = n + 1
    #             break
    #         else:
    #             node = paths.pop(0)[0]
    #             allowed_edges = list(filter(lambda e: node in e, edges))
    #             edges = list(set(edges).difference(set(allowed_edges)))
    #             for edge in allowed_edges:
    #                 i, j = nodes.index(edge[0]), nodes.index(edge[1])
    #                 A[i, j] = n
    #                 A[j, i] = n
    #                 n += 1
    #     for i in range(len(nodes)):
    #         A[i, i] = 0
    #     print("Number of edges: " + str(n) + " (for filtration cutoff)")
    #     return A, n, k

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




    # Gives a list of lists of barcode intervals in various dimensions
    # a word of warning: only include a dimension of 3
    def get_barcode(self, filt_mat, maxdim=2):
        b = ripser(filt_mat, distance_matrix=True, maxdim=maxdim)['dgms']
        return list(zip(range(maxdim+1), b))


    # calculates "barcodeyness" of a set of barcode intervals
    # essentially, feed in one dimension of the persistence intervals, as well
    # as a length by which to divide everything, and it'll see how sparse things
    # are, as well as report back with total number of introduced cycles
    def barcodeyness(self, intervals, length):
        tot = 0 # total normed persistence length
        num = len(intervals) # number of cycles in this dimension
        num_inf = 0 # number of infinitely-persisting cycles
        for intr in intervals:
            if np.isinf(intr[1]):
                num_inf += 1
                tot += (length - intr[0])/length
            else:
                tot += (intr[1] - intr[0])/length
        return num, tot/(num if num != 0 else 1), num_inf


    # plots a barcode; one of these days I'm gonna figure out how to pipeline Julia in here
    # and that way I'll be able to avoid opening up Eirene
    def plot_barcode(self, bars, dims=[1,2], length=None, k=None, labels=None):
        plt.figure(figsize=(12/2.54, 4), dpi=300)
        colors = ['b', 'c', 'g'] #necessary
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

    # Plots the barcode for the expositional ordering network
    def plot_expos_barcode(self, dims=[1,2], labels='edge'):
        # if self.ph_dict exists, it'll use that; otherwise, it'll make it
        bars, _ = self.get_barcode(dims=dims)
        self.plot_barcode(bars, dims=dims, labels=labels)

    """
    # saves a rivet bifiltration input file (of type II, see http://rivet.online/doc/input-data/)
    def make_rivet(self):
        file = open('rivet.txt', 'w')
        file.write('metric')
        file.write('cooc significance threshold')
        file.write

        file.close()
    """


