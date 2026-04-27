import pandas as pd
from typing import Dict
from imblearn.pipeline import Pipeline as ImbPipeline

data_storage: Dict[str, pd.DataFrame] = {}
job_storage: Dict[str, ImbPipeline] = {}