from src.data_utils import load_data, prepare, validate_data
def test_dataset_loads():
    df=load_data(); assert len(df)==64000
    assert set(["segment","visit","conversion","spend"]).issubset(df.columns)
def test_validation():
    r=validate_data(load_data()); assert r["valid_treatments"]
def test_prepare():
    d=prepare(load_data()); assert d.isna().sum().sum()==0
