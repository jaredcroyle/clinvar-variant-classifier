import pandas as pd

def load_and_clean_data(filepath: str, nrows: int = 200000) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t", nrows=nrows)
    df = df.dropna(subset=["ClinicalSignificance", "GeneSymbol"])
    df = df[df["ClinicalSignificance"].isin(["Pathogenic", "Benign"])]
    df["label"] = df["ClinicalSignificance"].map({"Pathogenic": 1, "Benign": 0})
    df["VariantLength"] = df["Stop"] - df["Start"]
    df["VariantLength"] = df["VariantLength"].fillna(0)
    df["Chromosome"] = df["Chromosome"].astype("category").cat.codes
    df["ReviewStatus"] = df["ReviewStatus"].astype("category").cat.codes
    df = df[["Chromosome", "VariantLength", "Start", "Stop", "PositionVCF", 
             "ReviewStatus", "NumberSubmitters", "SubmitterCategories", 
             "Type", "label"]]
    df = df.dropna()
    return df