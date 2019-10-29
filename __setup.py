#!/usr/bin/env python
# coding: utf-8

# # Library

# In[30]:

from __future__ import print_function
import pandas as pd
import numpy as np
import missingno as msno
import ipywidgets as widgets
from IPython.display import Markdown, display
# import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random
import pickle
from pandas2arff import *
from numpy import nan
from IPython.display import Markdown, HTML, display
from tabulate import tabulate
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets.embed import embed_minimal_html
from functools import partial
import itertools
from pandas2arff import *
from imblearn.over_sampling import RandomOverSampler

import os
import tempfile
import weka.core.jvm as jvm
import pygraphviz
import PIL
from PIL import Image
from weka.core.converters import Loader
# import weka.core.ContingencyTables.entropy
from weka.classifiers import Classifier, PredictionOutput, Evaluation
import weka.plot.graph as graph  # NB: pygraphviz and PIL are required
from weka.core.converters import Loader, Saver
import traceback
import weka.core.serialization as serialization
import weka.plot.graph as plot_graph
import weka.plot.classifiers as plot_cls

