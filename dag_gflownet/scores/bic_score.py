from math import lgamma, log
import sys

import numpy as np
from scipy.special import gammaln

from pgmpy.estimators import BaseEstimator
from collections import namedtuple

StateCounts = namedtuple('StateCounts', ['key', 'counts'])
LocalScore = namedtuple('LocalScore', ['key', 'score', 'prior'])

class StructureScore(BaseEstimator):
    """
    Abstract base class for structure scoring classes in pgmpy. Use any of the derived classes
    K2Score, BDeuScore, BicScore or AICScore. Scoring classes are
    used to measure how well a model is able to describe the given data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    Reference
    ---------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3
    """

    def __init__(self, data, **kwargs):
        super(StructureScore, self).__init__(data, **kwargs)

    def score(self, model):
        """
        Computes a score to measure how well the given `BayesianNetwork` fits
        to the data set.  (This method relies on the `local_score`-method that
        is implemented in each subclass.)

        Parameters
        ----------
        model: BayesianNetwork instance
            The Bayesian network that is to be scored. Nodes of the BayesianNetwork need to coincide
            with column names of data set.

        Returns
        -------
        score: float
            A number indicating the degree of fit between data and model

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.estimators import K2Score
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list('AB'))
        >>> data['C'] = data['B']
        >>> K2Score(data).score(BayesianNetwork([['A','B'], ['A','C']]))
        -24242.367348745247
        >>> K2Score(data).score(BayesianNetwork([['A','B'], ['B','C']]))
        -16273.793897051042
        """

        score = 0
        for node in model.nodes():
            score += self.local_score(node, model.predecessors(node))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model):
        """A (log) prior distribution over models. Currently unused (= uniform)."""
        return 0

    def structure_prior_ratio(self, operation):
        """Return the log ratio of the prior probabilities for a given proposed change to the DAG.
        Currently unused (=uniform)."""
        return 0


class BICScore(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with
    Dirichlet priors.  The BIC/MDL score ("Bayesian Information Criterion",
    also "Minimal Descriptive Length") is a log-likelihood score with an
    additional penalty for network complexity, to avoid overfitting.  The
    `score`-method measures how well a model is able to describe the given
    data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 802)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, **kwargs):

        self.num_variables = data.shape[1]
        print(self.num_variables)

        super(BICScore, self).__init__(data, **kwargs)
        
    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data
                local_score_before, local_score_after = self.get_local_scores(
                    target, indices, indices_after=indices_after)

                out_queue.put((True, *local_score_after))
                if local_score_before is not None:
                    out_queue.put((True, *local_score_before))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None, None))

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        sample_size = len(self.data)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score

    def get_local_scores(self, target, indices, indices_after=None):
        #asia
        fixed_order = ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']

        #child
        #fixed_order = ['BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray' ,'Grunting', 'LVHreport', 'LowerBodyO2' ,'RUQO2', 'CO2Report' ,'XrayReport' ,'Disease', 'GruntingReport' ,'Age' ,'LVH', 'DuctFlow', 'CardiacMixing' ,'LungParench', 'LungFlow' ,'Sick']

        #sports
        #fixed_order = ['RDlevel', 'possession', 'HTshots', 'ATshots', 'HTshotOnTarget', 'ATshotsOnTarget', 'HTgoals', 'ATgoals', 'HDA']

        #alarm
        #fixed_order = ['HISTORY', 'HREKG', 'LVFAILURE', 'ERRLOWOUTPUT', 'HRSAT', 'VENTALV', 'FIO2', 'VENTLUNG', 'STROKEVOLUME', 'LVEDVOLUME', 'BP', 'CO', 'HYPOVOLEMIA', 'INTUBATION', 'TPR', 'VENTMACH', 'CATECHOL', 'PULMEMBOLUS', 'MINVOL', 'CVP', 'INSUFFANESTH', 'HRBP', 'SAO2', 'HR', 'PRESS', 'ERRCAUTER', 'PVSAT', 'VENTTUBE', 'KINKEDTUBE', 'DISCONNECT', 'MINVOLSET', 'ANAPHYLAXIS', 'EXPCO2', 'ARTCO2', 'PCWP', 'SHUNT', 'PAP']
        
        #sachs
        #fixed_order = ['Akt', 'Erk', 'Jnk', 'Mek' ,'P38' ,'PIP2' ,'PIP3', 'PKA', 'PKC' ,'Plcg', 'Raf']

        #hailfinder
        '''
        fixed_order = [
        'N0_7muVerMo', 'SubjVertMo', 'QGVertMotion', 'CombVerMo', 'AreaMeso_ALS',
        'SatContMoist', 'RaoContMoist', 'CombMoisture', 'AreaMoDryAir', 'VISCloudCov',
        'IRCloudCover', 'CombClouds', 'CldShadeOth', 'AMInstabMt', 'InsInMt',
        'WndHodograph', 'OutflowFrMt', 'MorningBound', 'Boundaries', 'CldShadeConv',
        'CompPlFcst', 'CapChange', 'LoLevMoistAd', 'InsChange', 'MountainFcst',
        'Date', 'Scenario', 'ScenRelAMCIN', 'MorningCIN', 'AMCINInScen',
        'CapInScen', 'ScenRelAMIns', 'LIfr12ZDENSd', 'AMDewptCalPl', 'AMInsWliScen',
        'InsSclInScen', 'ScenRel3_4', 'LatestCIN', 'LLIW', 'CurPropConv',
        'ScnRelPlFcst', 'PlainsFcst', 'N34StarFcst', 'R5Fcst', 'Dewpoints',
        'LowLLapse', 'MeanRH', 'MidLLapse', 'MvmtFeatures', 'RHRatio',
        'SfcWndShfDis', 'SynForcng', 'TempDis', 'WindAloft', 'WindFieldMt',
        'WindFieldPln'
        ]
        '''

        #win95pts
        '''
        fixed_order = [
            "AppOK", "DataFile", "AppData", "DskLocal", "PrtSpool", "PrtOn", "PrtPaper", 
            "NetPrint", "PrtDriver", "PrtThread", "EMFOK", "GDIIN", "DrvSet", "DrvOK", 
            "GDIOUT", "PrtSel", "PrtDataOut", "PrtPath", "NtwrkCnfg", "PTROFFLINE", "NetOK", 
            "PrtCbl", "PrtPort", "CblPrtHrdwrOK", "LclOK", "DSApplctn", "PrtMpTPth", "DS_NTOK", 
            "DS_LCLOK", "PC2PRT", "PrtMem", "PrtTimeOut", "FllCrrptdBffr", "TnrSpply", "PrtData", 
            "Problem1", "AppDtGnTm", "PrntPrcssTm", "DeskPrntSpd", "PgOrnttnOK", "PrntngArOK", 
            "ScrnFntNtPrntrFnt", "CmpltPgPrntd", "GrphcsRltdDrvrSttngs", "EPSGrphc", "NnPSGrphc", 
            "PrtPScript", "PSGRAPHIC", "Problem4", "TrTypFnts", "FntInstlltn", "PrntrAccptsTrtyp", 
            "TTOK", "NnTTOK", "Problem5", "LclGrbld", "NtGrbld", "GrbldOtpt", "HrglssDrtnAftrPrnt", 
            "REPEAT", "AvlblVrtlMmry", "PSERRMEM", "TstpsTxt", "GrbldPS", "IncmpltPS", "PrtFile", 
            "PrtIcon", "Problem6", "Problem3", "PrtQueue", "NtSpd", "Problem2", "PrtStatPaper", 
            "PrtStatToner", "PrtStatMem", "PrtStatOff"
        ]
        '''
        #formed
        #fixed_order = ['Impulsivity', 'Intelligence', 'AbilityToCope', 'FinancialDifficulties', 'ProblematicLifeEvents', 'Victimisation', 'GangMember', 'AngerPT', 'ViolentThoughts', 'BPD', 'ASPD', 'AbuseNeglectAsChild', 'Violence', 'TimeAtRisk', 'Education', 'EmploymentOrTraining', 'AnxietyPT', 'DepressionPT', 'LivingCircumstances', 'SocialWithdraw', 'CriminalFamilyBackground', 'CriminalNetwork', 'ComplianceWithSupervision', 'NegativeAttitude', 'CriminalAttitude', 'SymptomsOfMentalIllness', 'ResponsivenessToTreatment', 'RefuseFailToAttendTherapy', 'HazardousDrinkingAfterRelease', 'CannabisBeforePrisonSentence', 'CocaineBeforePrisonSentence', 'EcstasyBeforePrisonSentence', 'EcstasyDuringPrisonSentence', 'CannabisDuringPrisonSentence', 'CocaineDuringPrisonSentence', 'CannabisAfterRelease', 'CocaineAfterRelease', 'EcstasyAfterRelease', 'EcstasyDependence', 'CannabisDependence', 'CocaineDependence', 'AnyDrugDependence', 'AlcoholDependence', 'DrugTreatment', 'AlcoholTreatment', 'AlcoholTreatmentGivenRFAT', 'DrugTreatmentGivenRFAT', 'HazardousDrinkingPT', 'EcstasyPT', 'CannabisPT', 'CocainePT', 'ResponseToTreatGivenAlcDep', 'ResponseToTreatGivenDrugDep', 'SubstanceUse', 'SubstanceMisuseDL', 'ViolentConvictions', 'ViolentConvRateGivenProt', 'Anger', 'AngerManagementGivenRFAT', 'AngerManagement', 'AttitudeDL', 'AggressionDL', 'MentalIllnessDL', 'SocialProtectiveObs', 'ViolentConvictionsRate', 'Depression', 'Anxiety', 'ThoughtInsertion', 'Hallucinations', 'StrangeExperiences', 'ParanoidDelusions', 'PsychiatricTreatment', 'PsychiatricTreatmentGivenRFAT', 'ThoughtInsertionPT', 'HallucinationsPT', 'StrangeExperiencesPT', 'ParanoidDelusionsPT', 'MentalIllnessSymptomCount', 'PCLRfacet3', 'Stress', 'PCLRfactor2', 'prioracq', 'Age', 'Gender', 'PriorViolentConvictions', 'pclrscore', 'DomesticStability', 'PCLRfactor1']

        #property
        #fixed_order = ['propertyManagement', 'propertyExpenses', 'actualRentalIncome', 'otherPropertyExpenses', 'rentalGrossProfit', 'rentalIncomeLoss%', 'rentalIncome', 'propertyPurchaseValue', 'propertyExpensesGrowth', 'rentalGrowth', 'stampDutyTaxBand', 'capitalGrowth', 'capitalGains', 'incomeTax', 'rentalNetProfitBeforeInterest', 'interestTaxRelief', 'interest', 'interestRate', 'borrowing', 'otherInterestFees', 'rentalGrossYield', 'rentalIncomeT+1', 'LTV', 'propertyValueT+1', 'stampDutyTax', 'otherPropertyExpensesT+1', 'netProfit']

        variable = fixed_order[target]

        if indices_after is None:
            parents = [fixed_order[index] for index in indices]
            score = self.local_score(variable, parents)
            local_score_after = LocalScore(key=(target, tuple(indices)), score=score, prior=0.0)
            local_score_before = None
        else:
            parents1 = [fixed_order[index] for index in indices]
            score1 = self.local_score(variable, parents1)
            local_score_before = LocalScore(key=(target, tuple(indices)), score=score1, prior=0.0)
            parents2 = [fixed_order[index] for index in indices_after]
            score2 = self.local_score(variable, parents2)
            local_score_after = LocalScore(key=(target, tuple(indices_after)), score=score2, prior=0.0)
        #print(local_score_before, local_score_after)
        return (local_score_before, local_score_after)