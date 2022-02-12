import logging
logfmt = '[%(levelname)s] %(asctime)s - %(message)s'
logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format=logfmt)

import pandas as pd
import numpy as np
#access feature computers
from cmcl.features.extract_constituents import CompositionTable

#access transformers
from cmcl.features.extract_categories import DummyTable
from cmcl.features.extract_categories import LabelGrouper

from cmcl.transforms.PCA import PCA

#access models
from cmcl.models.RFR import RFR

# feature accessors should saveable in some way for future use/reporting

# model accessors perhaps can be saved, but are often not.
# expensive models should be saved, though...

@pd.api.extensions.register_dataframe_accessor("ft")
class FeatureAccessor():
    """
    Conveniently and robustly define and retrieve feature tables

    when table does not exist, it will be created.

    basic chemical descriptors:
    composition_tbl = CmclFrame.ft.comp()
    prop_tbl = CmclFrame.ft.prop()
    
    pymatgen descriptors:
    matminer_tbl = CmclFrame.ft.mtmr()

    MEGnet descriptors:
    meg_tbl = CmclFrame.meg()
    """

    def __init__(self, df):
        self._validate(df)
        #getting original
        self._df = df
        #getting dummies 
        self._ohe_mod_df = None
        self._ohe_cols = None
        #getting physical comp features
        self._comp_mod_df = None
        self._comp_cols = None
        
    @staticmethod
    def _validate(df):
        """
        verify df contains
        a series of Formula
        at least one series of measurements
        """
        #notice: this sort of specifc validation should be done by the respective featurizers
        # only general checks should be done here
        pass
        #         if df.columns.values.shape[0] < 2:
        #             pass #warn user no ml can be done on current data
        #         else:
        #             pass
        #         if "Formula" not in df.columns:
        #             if "formula" in df.columns:
        #                 df.rename(columns = {"formula":"Formula"})
        #             else:
        #                 raise AttributeError("No Formula Column Named")

    def base(self):
        return self._df
        
    def _make_ohe(self):
        extender = DummyTable(self._df)
        self._ohe_cols = extender.make_and_get()
        self._ohe_mod_df = extender.df

    def _get_ohe(self):
        return self._ohe_mod_df[self._ohe_cols]

    def ohe(self):
        if self._ohe_cols is None:
            self._make_ohe()
        return self._get_ohe()

    def _make_comp(self):
        """get array of formula's constituent quantities"""
        extender = CompositionTable(self._df)
        self._comp_cols = extender.make_and_get()
        self._comp_mod_df = extender.df

    def _get_comp(self):
        return self._comp_mod_df[self._comp_cols]

    def comp(self):
        """
        once called, the resulting training set is static?
        note to self: maybe calling the accessor on new records added to the dataframe resets it...
        """
        if self._comp_cols is None:
            self._make_comp()
        return self._get_comp()

    
    
    def mtmr(self):
        """get array of dscribe inorganic crystal properties"""
        print("matminer Not Implemented")

    def meg(self):
        """get array of MEGnet properties"""
        print("megnet Not Implemented")


@pd.api.extensions.register_dataframe_accessor("ABX")
class PerovskiteAccessor():
    """
    Conveniently and robustly define and retrieve categorical summaries of data

    when table does not exist, it will be created.

    these summary columns are all intended to work with X.ft.ohe()
    which can one-hot encode any one in a simple oneliner

    X.sm.mix().ft.ohe()

    Human descriptors:
    mixing_sites = CmclFrame.ft.comp().sm.mix

    continuous metric descretizations
    distribution_table = CmclFrame["metric"].sm.bin(nbins)
    """
    def __init__(self, df):
        self._validate(df)
        #getting original
        self._df = df
        #getting mix categories
        self._mix_mod_df = None
        self._mix_cols = None
        
    @staticmethod
    def _validate(df):
        """
        verify df contains at least one series of measurements
        """
        pass

    def base(self):
        return self._df
        
    def _mixreader(self, row):
      mixstring = " & "
      stringlist=[]
      if row[0] != 1:
         stringlist.append("A")
      if row[1] != 1:
         stringlist.append("B")
      if row[2] != 1:
         stringlist.append("X")
      if stringlist:
         stringlist[-1] = stringlist[-1] + "-site"
      if not stringlist:
         stringlist.append("Pure")
      mixstring = mixstring.join(stringlist)
      return mixstring

    def mix(self):
        """
        provides default access to ColumnGrouper. categorizes
        perovskite's by site mixing when applied to a composition
        table.
        """
        segments = {"A":["MA", "FA", "Cs", "Rb", "K"],
                    "B":["Pb", "Sn", "Ge", "Ba", "Sr", "Ca", "Be", "Mg", "Si", "V", "Cr", "Mn", "Fe", "Ni", "Zn", "Pd", "Cd", "Hg"],
                    "X":["I", "Br", "Cl"]}
        extender = LabelGrouper(self._df, **segments)
        mixlog = extender.sum_groups()
        return mixlog.apply(lambda row: self._mixreader(row), axis=1)
                
@pd.api.extensions.register_dataframe_accessor("tf")
class TransformAccessor():
    """
    Conveniently and robustly define and retrieve transforms of tables

    when transform does not exist, it will be created.

    these transforms can be applied to any X.ft.function()
    with a simple one-liner
    
    X.ft.comp().ft.ohe()

    provides signal analysis transforms
    FFT, Hilbert, etc

    decompositions
    PCA, etc

    kernel transforms
    TSNE, UMAP, SISSO, etc
    """
    def __init__(self, df):
        self._validate(df)
        #getting original
        self._df = df
        #getting mix categories
        self._PCA_mod_df = None
        self._PCA_cols = None
        
    @staticmethod
    def _validate(df):
        """
        verify df contains numerical data of the float dtype
        """
        pass

    def base(self):
        return self._df
        
    def _make_pca(self):
        extender = PCATable(self._df)
        self._pca_cols = extender.make_and_get()
        self._pca_mod_df = extender.df

    def _get_pca(self):
        return self._pca_mod_df[self._pca_cols]
        
    def pca(self):
        #wishlist: make the score matrix recoverable for biplotting
        if self._pca_cols is None:
            self._make_pca()
        return self._get_pca()

@pd.api.extensions.register_dataframe_accessor("model")
class ModelAccessor():
    """
    Conveniently and robustly define and retrieve models of tables based on other tables

    An accessed table will have models created for it. models are
    stored with the accessor instance, predictions are not. 

    predictions and models can be obtained with a simple one-liner
    
    Y.model.RFR(X, optimize=True)

    provides
    RFR
    GRR
    NN
    """
    def __init__(self, Y):
        self._validate(Y)
        #getting original
        self._df = Y
        #existing models
        self._RFR = None
        
    @staticmethod
    def _validate(Y):
        """
        verify Y contains numerical data

        flag weather categorical or continuous?
        """
        pass

    def base(self):
        return self._df
        
    #def _parametrize_RFR(self, ntrees=100, max_features="sqrt", optimize=False):
    #    self._RFR = extender.r
    #    # take arguments from user, somehow, to customize the fit/interrogate the predictor
    #    pass
    # maybe... no sure how best to do optimization

    def _do_RFR(self, X, r=None, t=0.20): #extend args?
        modeler = RFR(X, self._df, t=t, r=r)
        modeler.train_test_return()
        self._RFR = modeler

    def RFR(self, X, r=None):
        """
        return a model of Y based on X, The form of X used,
        and optionally the model used to get Y.

        Both Y and X are returned with pandas multindex

        this can be used to access the train/test split for each
        dataframe conveniently using pandas tuple indexing, the pandas
        IndexSlice module, or the .xs (cross section) method.
        """
        if self._RFR is None:
            #self._parametrize_RFR()            
            self._do_RFR(X, r=r, t=0.20)
        return self._RFR.Y_stack, self._RFR.X_stack, self._RFR.r
