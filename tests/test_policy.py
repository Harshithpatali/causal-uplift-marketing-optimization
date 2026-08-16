from src.policy import recommend
def test_policy_positive_mens(): assert recommend(.10,.02,0)=="Mens E-Mail"
def test_policy_positive_womens(): assert recommend(.01,.05,0)=="Womens E-Mail"
def test_policy_no_email(): assert recommend(-.01,-.02,0)=="No E-Mail"
def test_threshold(): assert recommend(.01,.02,.03)=="No E-Mail"
