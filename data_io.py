import netCDF4

def read_input_nc(filename):
    """
    This reads the input filename and populates the data dictionaries.

    Copied from original Raysca
    """
    groups = ['medium','instrument','source','boundary']
    medium = {}
    instrument = {}
    source = {}
    boundary = {}
    with netCDF4.Dataset(filename) as ds:
        for group in groups:
            variables = list(ds[group].variables.keys())
            for var in variables:
                exec(group + "['%s'] = ds['%s']['%s'][:].data" % (var,group,var))
    return (medium, instrument, source, boundary)
