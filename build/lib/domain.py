import polytope as pc # It is also recommended to import cvxopt, which is a non-required dependency of polytope

def get_domain_bounding_box(config):
    data_bounds_list = config.data_bounds
    return [data_bounds_list[i:i + 2] for i in range(0, len(data_bounds_list), 2)]

def get_domain_polytope(config):
    data_bounding_box = get_domain_bounding_box(config)
    return pc.box2poly(data_bounding_box)