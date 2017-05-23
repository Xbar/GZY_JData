#!/usr/bin/env python

from genetic import *
from base import *

train_data = pd.read_hdf('gene_train')
train_label = pd.read_hdf('gene_label')

gc = GeneticCoder(6, 200)
scores = gc.train(train_data, train_label, 100)

np.save('genetic_result', gc.population)

