"Defines matplotlib stylesheet"
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# %%--  Matplotlib style sheet
mpl.style.use('seaborn-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] ='STIXGeneral'
mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.default'] = 'rm'
mpl.rcParams['mathtext.fallback'] = 'cm'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.grid.which']='both'
mpl.rcParams['axes.xmargin']=0.05
mpl.rcParams['axes.ymargin']=0.05
mpl.rcParams['grid.linewidth']= 0
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['figure.figsize'] = (8.09,5)
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['figure.dpi'] = 75
mpl.rcParams['image.cmap'] = "viridis"
mpl.rcParams['savefig.dpi'] = 150
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.viridis([0.8,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
mpl.rcParams['axes.titlesize'] = 16
# %%-
