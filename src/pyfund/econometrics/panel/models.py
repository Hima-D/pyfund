import warnings

from linearmodels.panel import PanelOLS, RandomEffects

warnings.filterwarnings("ignore")


class PanelModels:
    @staticmethod
    def fixed_effects(df, dep, exog):
        df = df.set_index(["entity", "date"])
        mod = PanelOLS(df[dep], df[exog], entity_effects=True, time_effects=True)
        return mod.fit(cov_type="clustered", cluster_entity=True)

    @staticmethod
    def random_effects(df, dep, exog):
        df = df.set_index(["entity", "date"])
        mod = RandomEffects(df[dep], df[exog])
        return mod.fit()
