'''
Previous experiment list and related functions
Old experiments were removed, i left just a sample of how to use the experiment list
'''
experiments = []

# Sample experiment, give arguments of experiment separated by -. Define how to treat arguments in main.
# experiments.append("QSTEP-64PATCH-boundary-INIT-harp-FOLD1-RES-E2D-halfaug-nobias-B200-RADAM-0.001-1000")  # one of the plots


def extract_param_from_experiment(experiment, char='-'):
    tokens = experiment.split(char)
    batch_size = None
    nepochs = None
    LR = None
    for t in tokens:
        if t[0] == 'B':
            try:
                batch_size = int(t[1:])
            except ValueError:
                raise ValueError("Problem with batch_size detection in token {}".format(t))
        elif '.' in t:
            try:
                LR = float(t)
            except ValueError as ve:
                raise ValueError("Experiment token contains '.' but its not a learning rate: {}".format(ve))
        elif t == "ADAM":
            print("ADAM experiment, using initial 0.0001 LR")
            LR = 0.0001
        else:
            try:
                nepochs = int(t)
            except ValueError:
                pass

    return batch_size, nepochs, LR


def check_name(word, basename, char='-'):
    '''
    Check presence of an argument in arg string
    '''
    return word in basename.split(char)


def spawn_folds(experiment, experiments, nfolds=5):
    '''
    Autmatically add fold specific experiments to list of experiments from one entry
    '''
    experiments.remove(experiment)

    middle_string = '{}FOLD'.format(nfolds)

    if '-{}-'.format(middle_string) in experiment:
        experiment = experiment.replace('-5FOLD-', '-')
    elif '{}-'.format(middle_string) in experiment:
        experiment = experiment.replace('-5FOLD-', '-')
    elif '-{}'.format(middle_string) in experiment:
        experiment = experiment.replace('-5FOLD-', '-')
    else:
        raise ValueError("Called spawn folds in an experiment without 5FOLD argument.")

    for i in range(1, nfolds + 1):
        experiments.append(experiment + '-FOLD{}'.format(i))


def check_experiment(experiment):
    '''
    Returns parameters for current experiment
    '''
    experiment_args = {}
    experiment_args["bias"] = not check_name("nobias", experiment)
    experiment_args["e2d"] = check_name("E2D", experiment)
    experiment_args["res"] = check_name("RES", experiment)
    experiment_args["small"] = check_name("SMALL", experiment)
    experiment_args["adam"] = check_name("ADAM", experiment)
    experiment_args["radam"] = check_name("RADAM", experiment)
    experiment_args["bn"] = not check_name("NOBN", experiment)
    experiment_args["dunet"] = check_name("DUNET", experiment)
    experiment_args["mixed"] = check_name("MIXED", experiment)
    experiment_args["mixharp"] = check_name("mixharp", experiment)
    experiment_args["ROTATIONINVARIANT"] = check_name("ROTATIONINVARIANT", experiment)
    experiment_args["oldharp"] = check_name("oldharp", experiment)
    experiment_args["harp"] = check_name("harp", experiment)
    experiment_args["volumetric"] = check_name("3D", experiment)
    experiment_args["aug3d"] = check_name("AUG", experiment) and experiment_args["volumetric"]
    experiment_args["patch32"] = check_name("32PATCH", experiment)
    experiment_args["dim"] = '3d' if check_name("E3DUNET", experiment) else '2d'
    experiment_args["e3d"] = check_name("E3D", experiment)
    experiment_args["hweights"] = check_name("HWEIGHTS", experiment)
    experiment_args["mni"] = check_name("MNI", experiment)
    experiment_args["multitask"] = check_name("MULTITASK", experiment)
    experiment_args["ce"] = check_name("CE", experiment)
    experiment_args["NLR"] = check_name("NLR", experiment)
    experiment_args["anyborder"] = check_name("ANYBORDER", experiment)
    experiment_args["softarget"] = check_name("SOFTINTG", experiment)
    experiment_args["psoftarget"] = check_name("pSOFTINTG", experiment)
    experiment_args["classify"] = check_name("CLASSIFY", experiment)
    experiment_args["newschedule"] = check_name("newschedule", experiment)
    experiment_args["center128"] = check_name("center128", experiment)
    experiment_args["center"] = check_name("center", experiment)
    experiment_args["noaug"] = check_name("noaug", experiment)
    experiment_args["bordersoft"] = check_name("bordersoft", experiment)
    experiment_args["affonly"] = check_name("affonly", experiment)
    experiment_args["intonly"] = check_name("intonly", experiment)
    experiment_args["noiseonly"] = check_name("noiseonly", experiment)
    experiment_args["sig5"] = check_name("sig5", experiment)
    experiment_args["sig10"] = check_name("sig10", experiment)
    experiment_args["sig15"] = check_name("sig15", experiment)
    experiment_args["gdl"] = check_name("GDL", experiment)
    experiment_args["center_halfaug"] = check_name("center_halfaug", experiment)
    experiment_args["center_fullaug"] = check_name("center_fullaug", experiment)
    experiment_args["halfaug"] = check_name("halfaug", experiment)
    experiment_args["patch64"] = check_name("64PATCH", experiment)
    experiment_args["INTNOFLIP"] = check_name("INTNOFLIP", experiment)
    experiment_args["NOINTNOFLIP"] = check_name("NOINTNOFLIP", experiment)
    experiment_args["INTFLIP"] = check_name("INTFLIP", experiment)
    experiment_args["QSTEP"] = check_name("QSTEP", experiment)
    experiment_args["4QSTEP"] = check_name("4QSTEP", experiment)
    experiment_args["NOSTEP"] = check_name("NOSTEP", experiment)
    experiment_args["INIT"] = check_name("INIT", experiment)
    experiment_args["3CH"] = check_name("3CH", experiment)
    experiment_args["ZB"] = check_name("ZB", experiment)
    experiment_args["sigaug"] = check_name("sigaug", experiment)
    experiment_args["sighalfaug"] = check_name("sighalfaug", experiment)
    experiment_args["boundary"] = check_name("boundary", experiment)
    experiment_args["NOP"] = check_name("NOP", experiment)
    experiment_args["FULLRANDOM"] = check_name("FULLRANDOM", experiment)
    experiment_args["DLBound"] = check_name("DLBound", experiment)
    experiment_args["batch_size"], experiment_args["nepochs"], experiment_args["LR"] = extract_param_from_experiment(experiment)

    experiment_args["FOLD"] = None
    for i in range(1, 6):
        if check_name("FOLD{}".format(i), experiment):
            experiment_args["FOLD"] = i

    return experiment_args


if __name__ == "__main__":
    print("Currently scheduled experiments:")
    for i, experiment in enumerate(experiments):
        print("{}: {}".format(i + 1, experiment))
