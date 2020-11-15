import os


def load_class(name):
    package = name.rsplit(".", 1)[0]
    klass = name.rsplit(".", 1)[1]
    mod = __import__(package, fromlist=[klass])
    return getattr(mod, klass)


def mk_parent_dir(f_name):
    if os.path.dirname(f_name) is not '' and not os.path.isdir(os.path.dirname(f_name)):
        os.makedirs(os.path.dirname(f_name))
