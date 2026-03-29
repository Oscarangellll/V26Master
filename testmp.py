

if __name__ == "__main__":
    
    import multiprocessing as mp
    mp.set_start_method("spawn") # Must be called excactly once
    
    from config import CaseConfig
    from optimization_models import ConsensusModelMP
    
    case = CaseConfig("cases/3W3B.yaml")

    model = ConsensusModelMP(case, [10, 11], {10: 0.5, 11: 0.5})
    model.optimize()

    for key, var in model.gamma_LT.items():
        print(key, var.X)
    for key, var in model.gamma_ST.items():
        print(key, var.X)

    print(model.ObjVal)
