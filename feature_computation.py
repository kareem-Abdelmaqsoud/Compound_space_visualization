
# coding: utf-8

# In[1]:


from __future__ import print_function
import time
import numpy as np
import pandas as pd


# In[2]:


# reading the data
df= pd.read_csv('data.csv', sep=";")
df.set_index("ChEMBL ID", inplace= True)
print(df.shape)
#df.head()


# In[3]:


df=df.drop(['Name','Synonyms','Type', 'Max Phase', 'Structure Type', 'QED Weighted','Passes Ro3', 'Inorganic Flag'], axis=1)


# In[4]:


df.replace(to_replace=["None"], value=np.nan, inplace=True)
df=df.dropna()
print(df.shape)


# In[5]:


#df.head()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import rdkit, rdkit.Chem, rdkit.Chem.Draw
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
import jax.numpy as jnp
import mordred, mordred.descriptors
import jax.experimental.optimizers as optimizers
import jax
import warnings
from zipfile import ZipFile
warnings.filterwarnings('ignore')
sns.set_context('notebook')
sns.set_style('dark',  {'xtick.bottom':True, 'ytick.left':True, 'xtick.color': '#666666', 'ytick.color': '#666666',
                        'axes.edgecolor': '#666666', 'axes.linewidth':     0.8 , 'figure.dpi': 300})
color_cycle = ['#1BBC9B', '#F06060', '#5C4B51', '#F3B562', '#6e5687']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_cycle) 
np.random.seed(0)


# In[7]:


# make object that can compute descriptors
calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
# make subsample from pandas df
molecules = [rdkit.Chem.MolFromSmiles(smi) for smi in df['Smiles']]

# view one molecule to make sure things look good.
#molecules[0]


# In[ ]:


# the invalid molecules were None, so we'll just
# use the fact the None is False in Python
valid_mol_idx = [bool(m) for m in molecules]
valid_mols = [m for m in molecules if m]


# In[53]:


from mordred import descriptors
from mordred import Calculator
from mordred.MoeType import EState_VSA
from mordred import MoeType
from mordred.AdjacencyMatrix import AdjacencyMatrix
from mordred import AcidBase
from mordred import BondCount
from mordred import DistanceMatrix
from mordred import HydrogenBond
from mordred import CarbonTypes
from mordred import Autocorrelation
from mordred import Polarizability
from mordred import MolecularDistanceEdge


# In[66]:


df = pd.DataFrame()
for i in [AdjacencyMatrix,MoeType, AdjacencyMatrix, AcidBase, BondCount, DistanceMatrix, HydrogenBond, CarbonTypes,Polarizability]:
    calc = Calculator(i)
    print(len(calc.descriptors))  # 1
    features= calc.pandas(valid_mols, nproc=10)
    df =pd.concat([df, features], axis=1)


# In[64]:


# we have some nans in features, likely because std was 0
df.dropna(inplace=True, axis=1)
print(f'We have {len(df.columns)} features per molecule')


# In[61]:


df.to_csv('comupted_features_1.csv', index=False)

