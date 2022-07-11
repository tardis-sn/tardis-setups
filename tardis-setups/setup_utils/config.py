import os


def find_file(name, path):
    for dirpath, _, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)
    return name


def config_modifier(conf):
    conf["montecarlo"]["nthreads"] = 1
    conf["montecarlo"]["last_no_of_packets"] = 1.0e5
    conf["montecarlo"]["no_of_virtual_packets"] = 10

    conf["atom_data"] = os.path.join(
        os.path.expanduser("~"), "Downloads", "tardis-data", conf["atom_data"]
    )

    if os.path.exists(conf["atom_data"]) == False:
        conf["atom_data"] = find_file(conf["atom_data"], "/")
        if os.path.exists(conf["atom_data"]) == False:
            raise

    return conf
